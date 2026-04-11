#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot reversal with 1d volume spike and 1d ADX trend filter
# - Long when price touches Camarilla L3 support with volume > 1.5x 20-period average and ADX > 25 (trending market)
# - Short when price touches Camarilla H3 resistance with volume > 1.5x 20-period average and ADX > 25
# - Uses Camarilla levels from 1d: H3/L3 as key intraday support/resistance levels
# - Volume confirmation ensures institutional participation
# - ADX filter ensures we trade in trending conditions to avoid whipsaws
# - Discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 20-40 trades/year (80-160 total over 4 years) to stay within limits
# - Works in bull (buying dips at L3) and bear (selling rallies at H3) markets

name = "4h_1d_camarilla_volume_adx_v1"
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
    
    # Load 1d data ONCE before loop for Camarilla, volume and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Extract 1d arrays
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla levels (based on previous day)
    # Camarilla: H3 = close + 1.1*(high-low)/6, L3 = close - 1.1*(high-low)/6
    camarilla_h3 = close_1d + 1.1 * (high_1d - low_1d) / 6
    camarilla_l3 = close_1d - 1.1 * (high_1d - low_1d) / 6
    
    # Calculate 1d ADX (14-period)
    # True Range
    tr1 = pd.Series(high_1d).shift(1) - pd.Series(low_1d).shift(1)
    tr2 = abs(pd.Series(high_1d).shift(1) - pd.Series(close_1d).shift(1))
    tr3 = abs(pd.Series(low_1d).shift(1) - pd.Series(close_1d).shift(1))
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    plus_dm = pd.Series(high_1d).diff()
    minus_dm = pd.Series(low_1d).diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    # Smoothed values
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean()
    plus_di_1d = 100 * (pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean() / atr_1d)
    minus_di_1d = 100 * (pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean() / atr_1d)
    dx_1d = 100 * abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = pd.Series(dx_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # 1d volume SMA (20-period)
    volume_series = pd.Series(volume_1d)
    volume_sma_20_1d = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(volume_sma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average (using 1d aligned volume)
        vol_confirm = volume_current > 1.5 * volume_sma_20_aligned[i]
        
        # ADX trend filter: trade only when ADX > 25 (trending market)
        adx_filter = adx_aligned[i] > 25
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: price touches L3 support with volume confirmation and ADX filter
        if price_low <= camarilla_l3_aligned[i] and vol_confirm and adx_filter:
            enter_long = True
        
        # Short: price touches H3 resistance with volume confirmation and ADX filter
        if price_high >= camarilla_h3_aligned[i] and vol_confirm and adx_filter:
            enter_short = True
        
        # Exit conditions: price moves back toward VWAP or opposite Camarilla touch
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price moves back above L3 or touches H3
            exit_long = (price_close > camarilla_l3_aligned[i]) or (price_high >= camarilla_h3_aligned[i])
        elif position == -1:
            # Exit short if price moves back below H3 or touches L3
            exit_short = (price_close < camarilla_h3_aligned[i]) or (price_low <= camarilla_l3_aligned[i])
        
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