#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot reversal with 1d volume confirmation and ADX trend filter
# - Camarilla levels from 1d: L3/H3 act as strong intraday support/resistance
# - Long when price closes back above L3 after touching it (mean reversion in trend)
# - Short when price closes back below H3 after touching it
# - Volume confirmation: current volume > 1.5x 24-period average (institutional participation)
# - ADX filter: only trade when ADX > 25 to avoid ranging markets and false reversals
# - Discrete position sizing: ±0.25 to control drawdown and minimize fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee limits for 12h
# - Works in bull (buy dips in uptrend) and bear (sell rallies in downtrend) markets

name = "12h_1d_camarilla_volume_adx_v1"
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
    
    # Load 1d data ONCE before loop for Camarilla, volume and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Extract 1d arrays
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate previous day's Camarilla levels (using prior day's OHLC)
    # Shift by 1 to use completed day's data only
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close[0] = np.nan  # First day has no previous
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    # Camarilla calculations
    range_1d = prev_high - prev_low
    camarilla_h3 = prev_close + range_1d * 1.1 / 6
    camarilla_l3 = prev_close - range_1d * 1.1 / 6
    camarilla_h4 = prev_close + range_1d * 1.1 / 2
    camarilla_l4 = prev_close - range_1d * 1.1 / 2
    
    # 1d volume SMA (24-period)
    volume_series = pd.Series(volume_1d)
    volume_sma_24_1d = volume_series.rolling(window=24, min_periods=24).mean().values
    
    # 1d ADX calculation (14-period)
    # True Range
    tr1 = pd.Series(high_1d).shift(1) - pd.Series(low_1d).shift(1)
    tr2 = abs(pd.Series(high_1d).shift(1) - pd.Series(close_1d).shift(1))
    tr3 = abs(pd.Series(low_1d).shift(1) - pd.Series(close_1d).shift(1))
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    dm_plus = pd.Series(high_1d).diff()
    dm_minus = -pd.Series(low_1d).diff()
    dm_plus = np.where((dm_plus > dm_minus) & (dm_plus > 0), dm_plus, 0)
    dm_minus = np.where((dm_minus > dm_plus) & (dm_minus > 0), dm_minus, 0)
    
    # Smoothed values
    atr_14 = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / np.where(atr_14 == 0, 1, atr_14)
    di_minus = 100 * dm_minus_smooth / np.where(atr_14 == 0, 1, atr_14)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / np.where((di_plus + di_minus) == 0, 1, (di_plus + di_minus))
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align 1d indicators to 12h timeframe
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    volume_sma_24_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_24_1d)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_l3_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or
            np.isnan(volume_sma_24_aligned[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        
        # Camarilla touch conditions (using wicks for touch detection)
        touched_l3 = price_low <= camarilla_l3_aligned[i]
        touched_h3 = price_high >= camarilla_h3_aligned[i]
        
        # Reversal conditions: price closes back inside the level after touching
        reversed_long = touched_l3 and price_close > camarilla_l3_aligned[i]
        reversed_short = touched_h3 and price_close < camarilla_h3_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 24-period average
        vol_confirm = volume_current > 1.5 * volume_sma_24_aligned[i]
        
        # ADX trend filter: only trade in trending markets (ADX > 25)
        trend_filter = adx_aligned[i] > 25
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: L3 touch and reversal + volume + trend
        if reversed_long and vol_confirm and trend_filter:
            enter_long = True
        
        # Short: H3 touch and reversal + volume + trend
        if reversed_short and vol_confirm and trend_filter:
            enter_short = True
        
        # Exit conditions: opposite Camarilla level break or volatility collapse
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price breaks below L4 (stronger support break)
            exit_long = price_close < camarilla_l4_aligned[i]
        elif position == -1:
            # Exit short if price breaks above H4 (stronger resistance break)
            exit_short = price_close > camarilla_h4_aligned[i]
        
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