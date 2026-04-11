#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(15) breakout + 1d volume spike + momentum filter
# - Donchian levels from 4h: upper/lower bands from last 15 periods
# - Long when price breaks above upper band with volume > 1.8x 20-period average (strong conviction)
# - Short when price breaks below lower band with volume > 1.8x 20-period average
# - Momentum filter: only trade when RSI(14) is between 30 and 70 to avoid overbought/oversold exhaustion
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 25-40 trades/year (100-160 total over 4 years) to stay within fee drag limits for 4h
# - Volume requirement (>1.8x average) ensures we only trade high-conviction breakouts
# - Momentum filter prevents entries at exhaustion points, improving win rate in both bull and bear markets

name = "4h_1d_donchian_volume_momentum_v1"
timeframe = "4h"
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
    
    # Load 1d data ONCE before loop for volume confirmation and momentum filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Pre-compute 1d volume SMA and RSI
    volume_1d = df_1d['volume'].values
    close_1d = df_1d['close'].values
    
    # 1d volume SMA (20-period)
    volume_series = pd.Series(volume_1d)
    volume_sma_20_1d = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # 1d RSI (14-period)
    delta = pd.Series(close_1d).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_14_1d = (100 - (100 / (1 + rs))).values
    
    # Align 1d indicators to 4h timeframe
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    rsi_14_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d)
    
    # Pre-compute 4h Donchian channels (15-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=15, min_periods=15).max().values
    donchian_lower = low_series.rolling(window=15, min_periods=15).min().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(volume_sma_20_aligned[i]) or np.isnan(rsi_14_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        
        # Donchian breakout conditions
        breakout_long = price_close > donchian_upper[i-1]  # Close above previous period's upper band
        breakout_short = price_close < donchian_lower[i-1]  # Close below previous period's lower band
        
        # Volume confirmation: current volume > 1.8x 20-period average (using 1d aligned volume)
        vol_confirm = volume_current > 1.8 * volume_sma_20_aligned[i]
        
        # Momentum filter: RSI between 30 and 70 to avoid exhaustion
        momentum_filter = (rsi_14_aligned[i] >= 30) & (rsi_14_aligned[i] <= 70)
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Donchian upper breakout + volume confirmation + momentum filter
        if breakout_long and vol_confirm and momentum_filter:
            enter_long = True
        
        # Short: Donchian lower breakdown + volume confirmation + momentum filter
        if breakout_short and vol_confirm and momentum_filter:
            enter_short = True
        
        # Exit conditions: opposite Donchian breakout or momentum exhaustion
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price breaks below lower band OR RSI > 70 (overbought)
            exit_long = (price_close < donchian_lower[i-1]) or (rsi_14_aligned[i] > 70)
        elif position == -1:
            # Exit short if price breaks above upper band OR RSI < 30 (oversold)
            exit_short = (price_close > donchian_upper[i-1]) or (rsi_14_aligned[i] < 30)
        
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