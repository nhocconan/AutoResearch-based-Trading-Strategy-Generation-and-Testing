#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h KAMA trend + 1d RSI mean reversion + 1w volume regime filter
# - Long: KAMA(12h) upward + RSI(1d) < 30 (oversold) + 1w volume > 1.5x 20-period avg (high conviction)
# - Short: KAMA(12h) downward + RSI(1d) > 70 (overbought) + 1w volume > 1.5x 20-period avg
# - Exit: RSI returns to 50 level or ATR-based stop (1.5 ATR)
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits
# - KAMA adapts to market noise, reducing whipsaw in ranging markets
# - RSI extremes provide mean-reversion edges in 12h timeframe
# - 1w volume filter ensures we trade only during periods of institutional participation

name = "12h_1d_1w_kama_rsi_volume_v1"
timeframe = "12h"
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
    
    # Load 1w data ONCE before loop for volume regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return signals
    
    # Pre-compute 1w volume SMA(20) for regime filter
    volume_1w = df_1w['volume'].values
    volume_sma_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_1w, volume_sma_20_1w)
    
    # Load 1d data ONCE before loop for RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return signals
    
    # Pre-compute 1d RSI(14)
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing (alpha = 1/14)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # Handle division by zero (when avg_loss == 0)
    rsi = np.where(avg_loss == 0, 100, rsi)
    
    # Align 1d RSI to 12h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Load 12h data ONCE before loop for KAMA
    # (Since primary timeframe is 12h, we compute KAMA directly from prices)
    # KAMA: Kaufman's Adaptive Moving Average
    # ER (Efficiency Ratio) = |net change| / sum(|changes|)
    # Smoothest ER = 0.1, Fastest ER = 1.0
    # SC = [ER * (fastest - slowest) + slowest]^2
    # KAMA = prev_KAMA + SC * (price - prev_KAMA)
    
    # Calculate ER over 10 periods
    change = np.abs(np.diff(close, prepend=close[0]))
    abs_change = np.abs(np.diff(close, prepend=close[0]))
    
    # Net change over 10 periods
    net_change = np.abs(close - np.roll(close, 10))
    net_change[:10] = 0  # Not enough data
    
    # Sum of absolute changes over 10 periods
    sum_abs_change = pd.Series(abs_change).rolling(window=10, min_periods=10).sum().values
    
    # Avoid division by zero
    er = np.where(sum_abs_change != 0, net_change / sum_abs_change, 0)
    
    # Smoothing constants
    fastest = 2 / (2 + 1)   # EMA(2)
    slowest = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fastest - slowest) + slowest) ** 2
    
    # Initialize KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # KAMA direction: 1 if rising, -1 if falling, 0 if flat
    kama_dir = np.where(kama > np.roll(kama, 1), 1, np.where(kama < np.roll(kama, 1), -1, 0))
    kama_dir[0] = 0
    
    # Pre-compute ATR for stoploss (12h timeframe)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(rsi_aligned[i]) or np.isnan(kama_dir[i]) or np.isnan(volume_sma_20_aligned[i]) or
            np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        
        # Indicator values
        rsi_current = rsi_aligned[i]
        kama_direction = kama_dir[i]
        
        # Volume regime filter: current volume > 1.5x 20-period average
        vol_regime = volume_current > 1.5 * volume_sma_20_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: KAMA up + RSI oversold + high volume regime
        if kama_direction == 1 and rsi_current < 30 and vol_regime:
            enter_long = True
        
        # Short: KAMA down + RSI overbought + high volume regime
        if kama_direction == -1 and rsi_current > 70 and vol_regime:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if RSI returns to 50 or ATR-based stop
            exit_long = (rsi_current >= 50) or (close_price <= entry_price - 1.5 * atr_14[i])
        elif position == -1:
            # Exit short if RSI returns to 50 or ATR-based stop
            exit_short = (rsi_current <= 50) or (close_price >= entry_price + 1.5 * atr_14[i])
        
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