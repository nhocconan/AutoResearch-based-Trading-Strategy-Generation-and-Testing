#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend + RSI mean reversion + volume spike. 
# Long when KAMA upward (bullish trend) + RSI < 30 (oversold) + volume > 1.5x 20-bar average.
# Short when KAMA downward (bearish trend) + RSI > 70 (overbought) + volume spike.
# Uses ATR trailing stop (2.0x) for risk management.
# Targets 30-100 total trades over 4 years (7-25/year) with discrete position sizing (0.25).
# KAMA adapts to market noise, reducing whipsaws in ranging markets while capturing trends.
# RSI extremes provide mean reversion entries aligned with higher-timeframe trend.

name = "1d_KAMA_Trend_RSI_MeanRev_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAMA ( Kaufman Adaptive Moving Average ) - 10-period ER, 2/30 SC
    def calculate_kama(close, er_period=10, fast_sc=2, slow_sc=30):
        change = np.abs(np.diff(close, prepend=close[0]))
        volatility = np.abs(np.diff(close)).rolling(window=er_period, min_periods=1).sum().values
        er = np.where(volatility != 0, change / volatility, 0)
        sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    # Load 1w data ONCE before loop for EMA50 trend filter (additional confirmation)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # KAMA for trend
    kama = calculate_kama(close, er_period=10, fast_sc=2, slow_sc=30)
    
    # RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=1).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    # ATR for trailing stop
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    start_idx = max(50, 14, 20)  # warmup for EMA50, KAMA, RSI, and volume MA
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if np.isnan(ema_50_aligned[i]) or np.isnan(kama[i]) or np.isnan(rsi[i]):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            continue
        
        # Trend filters: KAMA direction + 1w EMA50 alignment
        kama_rising = kama[i] > kama[i-1]
        kama_falling = kama[i] < kama[i-1]
        price_above_1w_ema = close[i] > ema_50_aligned[i]
        price_below_1w_ema = close[i] < ema_50_aligned[i]
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_atr = atr[i]
        curr_volume_confirm = volume_confirm[i]
        curr_rsi = rsi[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: KAMA rising + price above 1w EMA50 + RSI < 30 (oversold) + volume confirmation
            if kama_rising and price_above_1w_ema and curr_rsi < 30 and curr_volume_confirm:
                signals[i] = 0.25
                position = 1
                highest_since_entry = curr_close
            # Short: KAMA falling + price below 1w EMA50 + RSI > 70 (overbought) + volume confirmation
            elif kama_falling and price_below_1w_ema and curr_rsi > 70 and curr_volume_confirm:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = curr_close
        
        elif position == 1:  # Long position
            # Update highest high since entry
            if curr_high > highest_since_entry:
                highest_since_entry = curr_high
            
            # Trailing stop: 2.0 * ATR below highest since entry
            if curr_close < highest_since_entry - 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Update lowest low since entry
            if curr_low < lowest_since_entry:
                lowest_since_entry = curr_low
            
            # Trailing stop: 2.0 * ATR above lowest since entry
            if curr_close > lowest_since_entry + 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals