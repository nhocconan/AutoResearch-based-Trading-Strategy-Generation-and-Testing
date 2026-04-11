#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend with RSI mean reversion and 1w volume spike filter
# - Long: KAMA upward (close > KAMA), RSI(14) < 30 (oversold), 1w volume > 1.5x 20-period average (institutional interest)
# - Short: KAMA downward (close < KAMA), RSI(14) > 70 (overbought), 1w volume > 1.5x 20-period average
# - Exit: RSI returns to 50 (mean reversion) or opposite KAMA crossover
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 15-25 trades/year (60-100 total over 4 years) to stay within fee drag limits
# - KAMA adapts to market noise, reducing whipsaw in ranging markets
# - RSI extremes provide mean reversion entries in trending markets
# - 1w volume spike confirms institutional participation, reducing false signals
# - Works in bull markets (buy oversold dips in uptrend) and bear markets (sell overbought rallies in downtrend)

name = "1d_1w_kama_rsi_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    # Load 1w data ONCE before loop for volume filter (MTF rule compliance)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return signals
    
    # Pre-compute 1w volume spike filter
    volume_1w = df_1w['volume'].values
    volume_sma_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    volume_spike_1w = volume_1w > 1.5 * volume_sma_20_1w
    volume_spike_aligned = align_htf_to_ltf(prices, df_1w, volume_spike_1w)
    
    # Pre-compute 1d KAMA (adaptive moving average)
    # Efficiency Ratio (ER)
    change = np.abs(np.diff(close, 1))
    change = np.insert(change, 0, 0)
    volatility = np.sum(np.abs(np.diff(close, 1)), axis=0) if False else None  # placeholder for correct calc
    # Correct volatility calculation: sum of absolute changes over ER period
    volatility = pd.Series(np.abs(np.diff(close, 1))).rolling(window=10, min_periods=1).sum().values
    volatility = np.insert(volatility, 0, 0)  # align length
    er = np.where(volatility > 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    # KAMA calculation
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Pre-compute 1d RSI(14)
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_ma = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    loss_ma = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(loss_ma > 0, gain_ma / loss_ma, 0)
    rsi = 100 - (100 / (1 + rs))
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(volume_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        
        # KAMA trend: close above/below KAMA
        kama_trend_up = close_price > kama[i]
        kama_trend_down = close_price < kama[i]
        
        # RSI conditions
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        rsi_neutral = (rsi[i] >= 40) & (rsi[i] <= 60)  # exit zone
        
        # Volume confirmation: 1w volume spike
        vol_confirm = volume_spike_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: KAMA up, RSI oversold, volume spike
        if kama_trend_up and rsi_oversold and vol_confirm:
            enter_long = True
        
        # Short: KAMA down, RSI overbought, volume spike
        if kama_trend_down and rsi_overbought and vol_confirm:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if RSI returns to neutral or KAMA turns down
            exit_long = rsi_neutral or kama_trend_down
        elif position == -1:
            # Exit short if RSI returns to neutral or KAMA turns up
            exit_short = rsi_neutral or kama_trend_up
        
        # Track entry price for stoploss calculation
        if enter_long or enter_short:
            entry_price = close_price
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals