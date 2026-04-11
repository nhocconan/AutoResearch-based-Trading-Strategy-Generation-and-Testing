#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend + RSI(2) extreme + 1w volume spike
# - Long: KAMA rising (bullish trend), RSI(2) < 10 (deep oversold), 1w volume > 2.0x 20-period avg
# - Short: KAMA falling (bearish trend), RSI(2) > 90 (deep overbought), 1w volume > 2.0x 20-period avg
# - Exit: RSI(2) returns to 50 level or opposite RSI extreme
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 20-50 trades/year (80-200 total over 4 years) to stay within fee drag limits
# - KAMA adapts to market noise, reducing whipsaw in choppy conditions
# - RSI(2) captures extreme short-term reversals effectively
# - 1w volume spike confirms institutional participation in the move

name = "1d_1w_kama_rsi2_volume_v1"
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
    
    # Load 1w data ONCE before loop for volume confirmation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return signals
    
    # Pre-compute 1w volume SMA(20)
    volume_1w = df_1w['volume'].values
    volume_sma_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_1w, volume_sma_20_1w)
    
    # Pre-compute KAMA(10) on 1d timeframe
    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(close, n=10))
    change = np.concatenate([[np.nan], change])  # align with original length
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0) if False else None  # placeholder
    # Recalculate volatility properly: sum of absolute daily changes over 10 periods
    volatility = np.zeros_like(close)
    for i in range(n):
        if i < 10:
            volatility[i] = np.nan
        else:
            volatility[i] = np.sum(np.abs(np.diff(close[i-10:i+1])))
    er = np.where(volatility != 0, change / volatility, 0)
    
    # Smoothing constants
    fastest = 2 / (2 + 1)  # EMA(2)
    slowest = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fastest - slowest) + slowest) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Pre-compute RSI(2) on 1d timeframe
    delta = np.diff(close)
    delta = np.concatenate([[np.nan], delta])  # align with original length
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing (alpha = 1/period)
    avg_gain = pd.Series(gain).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # Handle division by zero (when avg_loss == 0)
    rsi = np.where(avg_loss == 0, 100, rsi)
    rsi = np.where(avg_gain == 0, 0, rsi)
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(kama[i]) or np.isnan(kama[i-1]) or np.isnan(rsi[i]) or 
            np.isnan(volume_sma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        
        # KAMA direction: rising if current > previous
        kama_rising = kama[i] > kama[i-1]
        kama_falling = kama[i] < kama[i-1]
        
        # RSI(2) values
        rsi_current = rsi[i]
        
        # Volume confirmation: current volume > 2.0x 20-period average
        vol_confirm = volume_current > 2.0 * volume_sma_20_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: KAMA rising + RSI(2) < 10 (deep oversold) + volume spike
        if kama_rising and rsi_current < 10 and vol_confirm:
            enter_long = True
        
        # Short: KAMA falling + RSI(2) > 90 (deep overbought) + volume spike
        if kama_falling and rsi_current > 90 and vol_confirm:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if RSI(2) returns to 50 or reaches overbought
            exit_long = (rsi_current >= 50) or (rsi_current > 80)
        elif position == -1:
            # Exit short if RSI(2) returns to 50 or reaches oversold
            exit_short = (rsi_current <= 50) or (rsi_current < 20)
        
        # Track entry price for reference (not used in stoploss, but kept for consistency)
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