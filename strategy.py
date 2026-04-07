#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour price action with 1-day Bollinger Band squeeze breakout and volume confirmation
# Long when price breaks above upper Bollinger Band (20,2) + volume > 1.5x average volume + price > 1-day EMA50
# Short when price breaks below lower Bollinger Band (20,2) + volume > 1.5x average volume + price < 1-day EMA50
# Exit when price returns to middle Bollinger Band or volatility drops (BB width < 30th percentile)
# Stoploss at 2.0 * ATR(14)
# Position size: 0.25 (25% of capital)
# Uses 1-day Bollinger Bands for volatility regime and trend filter
# Target: 100-200 total trades over 4 years (25-50/year)

name = "4h_bb_squeeze_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-day data for Bollinger Bands and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1-day Bollinger Bands (20, 2)
    close_1d = df_1d['close'].values
    close_1d_s = pd.Series(close_1d)
    bb_mid = close_1d_s.rolling(window=20, min_periods=20).mean().values
    bb_std = close_1d_s.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_width = bb_upper - bb_lower
    
    # Calculate percentile of BB width for volatility regime (30th percentile = low volatility threshold)
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # 1-day EMA50 for trend filter
    ema50_1d = close_1d_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1-day indicators to 4h
    bb_upper_aligned = align_htf_to_ltf(prices, df_1d, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1d, bb_lower)
    bb_mid_aligned = align_htf_to_ltf(prices, df_1d, bb_mid)
    bb_width_percentile_aligned = align_htf_to_ltf(prices, df_1d, bb_width_percentile)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 4-period RSI for entry confirmation (optional filter)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_s = pd.Series(gain)
    loss_s = pd.Series(loss)
    avg_gain = gain_s.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = loss_s.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 4-period ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Average volume for volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(bb_upper_aligned[i]) or np.isnan(bb_lower_aligned[i]) or 
            np.isnan(bb_mid_aligned[i]) or np.isnan(bb_width_percentile_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price returns to middle Bollinger Band or low volatility regime
            elif close[i] >= bb_mid_aligned[i] or bb_width_percentile_aligned[i] < 30:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price returns to middle Bollinger Band or low volatility regime
            elif close[i] <= bb_mid_aligned[i] or bb_width_percentile_aligned[i] < 30:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Bollinger Band breakout with volume confirmation and trend filter
            # Volume filter: volume > 1.5x average volume
            volume_confirm = volume[i] > 1.5 * vol_ma[i]
            
            # Long: price breaks above upper BB + volume confirmation + price > 1-day EMA50 (uptrend)
            if close[i] > bb_upper_aligned[i] and volume_confirm and close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below lower BB + volume confirmation + price < 1-day EMA50 (downtrend)
            elif close[i] < bb_lower_aligned[i] and volume_confirm and close[i] < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals