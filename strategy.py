#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d volume spike and 1d choppiness regime filter
# - Long: Price breaks above H3 Camarilla level (from prior 1d) + volume > 1.5x 20-period avg + CHOP(14) < 61.8 (trending)
# - Short: Price breaks below L3 Camarilla level (from prior 1d) + volume > 1.5x 20-period avg + CHOP(14) < 61.8 (trending)
# - Exit: Price returns to Camarilla pivot point (PP) or ATR-based stop (2.0 ATR)
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits
# - Camarilla pivots provide precise intraday support/resistance levels
# - Volume spike confirms institutional participation in breakouts
# - Choppiness filter ensures we only trade in trending regimes to avoid whipsaw

name = "12h_1d_camarilla_breakout_volume_chop_v1"
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
    
    # Load 1d data ONCE before loop for Camarilla pivots, volume, and choppiness
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Pre-compute 1d OHLC for Camarilla calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for prior 1d bar
    # PP = (High + Low + Close) / 3
    # R4 = Close + ((High - Low) * 1.1 / 2)
    # R3 = Close + ((High - Low) * 1.1 / 4)
    # R2 = Close + ((High - Low) * 1.1 / 6)
    # R1 = Close + ((High - Low) * 1.1 / 12)
    # S1 = Close - ((High - Low) * 1.1 / 12)
    # S2 = Close - ((High - Low) * 1.1 / 6)
    # S3 = Close - ((High - Low) * 1.1 / 4)
    # S4 = Close - ((High - Low) * 1.1 / 2)
    # We use H3 = R3 and L3 = S3 for breakouts
    pp = (high_1d + low_1d + close_1d) / 3.0
    r3 = close_1d + ((high_1d - low_1d) * 1.1 / 4.0)
    s3 = close_1d - ((high_1d - low_1d) * 1.1 / 4.0)
    
    # Align 1d Camarilla levels to 12h timeframe (use prior completed 1d bar)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Pre-compute 1d volume confirmation (20-period average)
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Pre-compute 1d Choppiness Index (CHOP) for regime filter
    # CHOP = 100 * log10(SUM(TR,14) / (MAX(HH,14) - MIN(LL,14))) / log10(14)
    tr_1d = np.maximum(high_1d - low_1d, np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), np.abs(low_1d - np.roll(close_1d, 1))))
    tr_1d[0] = high_1d[0] - low_1d[0]
    
    # True Range sum over 14 periods
    tr_sum_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    
    # Highest High and Lowest Low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index
    chop_denominator = hh_14 - ll_14
    chop_denominator = np.where(chop_denominator == 0, 1, chop_denominator)  # avoid division by zero
    chop_raw = 100 * np.log10(tr_sum_14 / chop_denominator) / np.log10(14)
    chop = np.where(np.isnan(chop_raw), 50, chop_raw)  # default to neutral if invalid
    
    # Align 1d Choppiness to 12h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Pre-compute ATR for stoploss (12h timeframe)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(pp_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(volume_sma_20_aligned[i]) or np.isnan(chop_aligned[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        
        # Camarilla levels
        pp_val = pp_aligned[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20_aligned[i]
        
        # Regime filter: CHOP < 61.8 indicates trending market (avoid choppy/ranging)
        trending_regime = chop_aligned[i] < 61.8
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long breakout: price breaks above H3 (R3) with volume and trend
        if close_price > r3_val and vol_confirm and trending_regime:
            enter_long = True
        
        # Short breakout: price breaks below L3 (S3) with volume and trend
        if close_price < s3_val and vol_confirm and trending_regime:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price returns to pivot point or ATR-based stop
            exit_long = (close_price <= pp_val) or (close_price <= entry_price - 2.0 * atr_14[i])
        elif position == -1:
            # Exit short if price returns to pivot point or ATR-based stop
            exit_short = (close_price >= pp_val) or (close_price >= entry_price + 2.0 * atr_14[i])
        
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