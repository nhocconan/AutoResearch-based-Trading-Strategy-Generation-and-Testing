#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 12h volume confirmation + ATR volatility filter
# - Donchian levels from 4h: upper/lower bands act as dynamic support/resistance
# - Long when price breaks above upper band with volume > 1.3x 20-period average
# - Short when price breaks below lower band with volume > 1.3x 20-period average
# - ATR filter: only trade when ATR(14) > 0.4 * ATR(50) to avoid low volatility chop
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 19-50 trades/year (75-200 total over 4 years) to stay within fee drag limits
# - Donchian breakouts work in both bull (breakouts with volume) and bear (breakdowns with volume) markets
# - 12h HTF provides reliable volume confirmation, 4h timeframe balances frequency and cost

name = "4h_12h_donchian_volume_atr_v1"
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
    
    # Load 12h data ONCE before loop for volume confirmation and ATR
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return signals
    
    # Pre-compute 12h volume SMA and ATR
    volume_12h = df_12h['volume'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True range for ATR
    tr1 = pd.Series(high_12h).shift(1) - pd.Series(low_12h).shift(1)
    tr2 = abs(pd.Series(high_12h).shift(1) - pd.Series(close_12h).shift(1))
    tr3 = abs(pd.Series(low_12h).shift(1) - pd.Series(close_12h).shift(1))
    tr_12h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14_12h = pd.Series(tr_12h).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_50_12h = pd.Series(tr_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 12h volume SMA (20-period)
    volume_series = pd.Series(volume_12h)
    volume_sma_20_12h = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Align 12h indicators to 4h timeframe
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_12h, volume_sma_20_12h)
    atr_14_aligned = align_htf_to_ltf(prices, df_12h, atr_14_12h)
    atr_50_aligned = align_htf_to_ltf(prices, df_12h, atr_50_12h)
    
    # Pre-compute 4h Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(volume_sma_20_aligned[i]) or np.isnan(atr_14_aligned[i]) or np.isnan(atr_50_aligned[i])):
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
        
        # Volume confirmation: current volume > 1.3x 20-period average (using 12h aligned volume)
        vol_confirm = volume_current > 1.3 * volume_sma_20_aligned[i]
        
        # ATR filter: trade only when short-term ATR > 0.4 * long-term ATR (avoid low volatility)
        atr_filter = atr_14_aligned[i] > 0.4 * atr_50_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Donchian upper breakout + volume confirmation + ATR filter
        if breakout_long and vol_confirm and atr_filter:
            enter_long = True
        
        # Short: Donchian lower breakdown + volume confirmation + ATR filter
        if breakout_short and vol_confirm and atr_filter:
            enter_short = True
        
        # Exit conditions: opposite Donchian breakout or volatility collapse
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price breaks below lower band OR volatility collapses
            exit_long = (price_close < donchian_lower[i-1]) or (not atr_filter)
        elif position == -1:
            # Exit short if price breaks above upper band OR volatility collapses
            exit_short = (price_close > donchian_upper[i-1]) or (not atr_filter)
        
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