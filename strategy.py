#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h/1d Bollinger Band squeeze + momentum breakout
# - Bollinger Bands (20,2) on 1h for squeeze detection and breakout signals
# - 1d RSI(14) trend filter to avoid counter-trend trades
# - Volume spike confirmation (>2x 20-period average) to filter false breakouts
# - Long when: BB squeeze + price breaks above upper band + 1d RSI > 50 + volume spike
# - Short when: BB squeeze + price breaks below lower band + 1d RSI < 50 + volume spike
# - Uses discrete position sizing: ±0.20 to limit drawdown and reduce fee churn
# - Target: 15-37 trades/year (60-150 total over 4 years) to stay within fee drag limits for 1h
# - Works in both bull (breakouts with volume) and bear (breakdowns with volume) markets
# - Bollinger squeeze identifies low volatility periods preceding explosive moves
# - 1d RSI filter ensures alignment with higher timeframe momentum

name = "1h_1d_bb_squeeze_momentum_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for RSI trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return signals
    
    # Pre-compute 1d RSI(14)
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Pre-compute Bollinger Bands on 1h data (20,2)
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_band = sma_20 + 2 * std_20
    lower_band = sma_20 - 2 * std_20
    
    # Bollinger Band width for squeeze detection
    bb_width = (upper_band - lower_band) / sma_20
    # Squeeze: BB width below 20-period average of BB width
    bb_width_ma = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    squeeze = bb_width < bb_width_ma
    
    # Pre-compute volume SMA (20-period)
    volume_series = pd.Series(volume)
    volume_sma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    
    for i in range(20, n):  # Start after 20-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(sma_20[i]) or np.isnan(std_20[i]) or np.isnan(bb_width[i]) or
            np.isnan(rsi_1d_aligned[i]) or np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        
        # Bollinger Band conditions
        breakout_up = price_high > upper_band[i]  # Break above upper band
        breakout_down = price_low < lower_band[i]  # Break below lower band
        
        # Volume confirmation: current volume > 2x 20-period average
        vol_confirm = volume_current > 2.0 * volume_sma_20[i]
        
        # 1d RSI trend filter
        rsi_bullish = rsi_1d_aligned[i] > 50
        rsi_bearish = rsi_1d_aligned[i] < 50
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: BB squeeze + breakout up + 1d RSI bullish + volume spike
        if squeeze[i] and breakout_up and rsi_bullish and vol_confirm:
            enter_long = True
        
        # Short: BB squeeze + breakout down + 1d RSI bearish + volume spike
        if squeeze[i] and breakout_down and rsi_bearish and vol_confirm:
            enter_short = True
        
        # Exit conditions: opposite breakout or loss of squeeze
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if breakdown below lower band OR loss of squeeze
            exit_long = (price_low < lower_band[i]) or (not squeeze[i])
        elif position == -1:
            # Exit short if breakout above upper band OR loss of squeeze
            exit_short = (price_high > upper_band[i]) or (not squeeze[i])
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.20
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.20
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals