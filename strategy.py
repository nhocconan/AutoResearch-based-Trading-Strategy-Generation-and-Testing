#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend + RSI(2) mean reversion + 1w volume regime filter
# - Long: KAMA(10,2,30) rising, RSI(2) < 10, 1w volume > 1.5x 20-period average
# - Short: KAMA falling, RSI(2) > 90, 1w volume > 1.5x 20-period average
# - Exit: RSI(2) > 50 for long, RSI(2) < 50 for short
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 15-25 trades/year (60-100 total over 4 years) to stay within fee drag limits
# - KAMA adapts to market noise, reducing whipsaw in ranging markets
# - RSI(2) captures extreme short-term mean reversion opportunities
# - 1w volume filter ensures we only trade during institutional participation

name = "1d_kama_rsi2_1w_volume_v1"
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
    
    # Load 1w data ONCE before loop for volume regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return signals
    
    # Pre-compute 1w volume SMA(20) for regime filter
    volume_1w = df_1w['volume'].values
    volume_sma_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_1w, volume_sma_20_1w)
    
    # Pre-compute 1d KAMA(10,2,30) for trend
    # ER = |Change| / Sum|Changes|
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = pd.Series(change).rolling(window=10, min_periods=10).sum().values
    direction = np.abs(np.diff(close, prepend=close[0]))
    er = np.where(volatility > 0, direction / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA calculation
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Pre-compute 1d RSI(2) for mean reversion signals
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(volume_sma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        
        # KAMA direction: rising if today > yesterday
        kama_rising = kama[i] > kama[i-1]
        kama_falling = kama[i] < kama[i-1]
        
        # RSI(2) extreme levels
        rsi_oversold = rsi[i] < 10
        rsi_overbought = rsi[i] > 90
        rsi_exit_long = rsi[i] > 50
        rsi_exit_short = rsi[i] < 50
        
        # Volume regime filter: 1w volume > 1.5x 20-period average
        vol_regime = volume_current > 1.5 * volume_sma_20_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: KAMA rising + RSI(2) oversold + volume regime
        if kama_rising and rsi_oversold and vol_regime:
            enter_long = True
        
        # Short: KAMA falling + RSI(2) overbought + volume regime
        if kama_falling and rsi_overbought and vol_regime:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if RSI(2) > 50 (mean reversion complete)
            exit_long = rsi_exit_long
        elif position == -1:
            # Exit short if RSI(2) < 50 (mean reversion complete)
            exit_short = rsi_exit_short
        
        # Track entry price for reference (not used in stoploss, but for consistency)
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