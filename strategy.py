#!/usr/bin/env python3
"""
1d_Williams_Alligator_Regime_Adaptive_v1
Hypothesis: Williams Alligator (jaw/teeth/lips) on daily timeframe defines trend regime.
In bull regime (lips > teeth > jaw): buy pullbacks to teeth with volume confirmation.
In bear regime (jaw > teeth > lips): sell rallies to teeth with volume confirmation.
In chop regime (Alligator sleeping): mean revert at Bollinger Bands (20,2) with volume filter.
Weekly trend filter (EMA34_1w) avoids counter-trend trades in strong weekly trends.
Designed for 1d timeframe to target 30-100 trades over 4 years (7-25/year).
Uses discrete sizing (0.25) and ATR-based stoploss (2.0x) for risk control.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for EMA34 trend filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # === 1w OHLC for EMA34 trend filter ===
    df_1w_close = df_1w['close'].values
    ema_34_1w = pd.Series(df_1w_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # === 1d OHLC for Williams Alligator (SMAs) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    df_1d_close = df_1d['close'].values
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_volume = df_1d['volume'].values
    
    # Williams Alligator: Jaw (13), Teeth (8), Lips (5) SMAs of median price
    median_price = (df_1d_high + df_1d_low) / 2
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    
    # Align Alligator lines to 1d timeframe (already 1d, but using align for consistency)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # === Bollinger Bands (20,2) for chop regime mean reversion ===
    close_s = pd.Series(df_1d_close)
    bb_mid = close_s.rolling(window=20, min_periods=20).mean().values
    bb_std = close_s.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    
    bb_upper_aligned = align_htf_to_ltf(prices, df_1d, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1d, bb_lower)
    
    # === ATR (14-period) for stoploss ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # === Volume confirmation (20-period average) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(60, n):
        # Skip if indicators not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) 
            or np.isnan(ema_34_1w_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        volume_now = volume[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        ema_1w_trend = ema_34_1w_aligned[i]
        vol_avg = vol_ma[i]
        bb_upper_val = bb_upper_aligned[i]
        bb_lower_val = bb_lower_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirmed = volume_now > 1.5 * vol_avg
        
        # Determine Alligator regime
        # Bull: lips > teeth > jaw
        # Bear: jaw > teeth > lips
        # Chop: otherwise (Alligator sleeping)
        is_bull = (lips_val > teeth_val) and (teeth_val > jaw_val)
        is_bear = (jaw_val > teeth_val) and (teeth_val > lips_val)
        is_chop = not (is_bull or is_bear)
        
        if position == 0:
            # No position - look for entries
            if is_bull:
                # Bull regime: buy pullbacks to teeth with volume
                long_condition = (price <= teeth_val * 1.005) and (price >= teeth_val * 0.995) and volume_confirmed
                # Weekly trend filter: avoid longing in strong weekly downtrend
                weekly_filter = price > ema_1w_trend
                if long_condition and weekly_filter:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
            elif is_bear:
                # Bear regime: sell rallies to teeth with volume
                short_condition = (price <= teeth_val * 1.005) and (price >= teeth_val * 0.995) and volume_confirmed
                # Weekly trend filter: avoid shorting in strong weekly uptrend
                weekly_filter = price < ema_1w_trend
                if short_condition and weekly_filter:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
            elif is_chop:
                # Chop regime: mean revert at Bollinger Bands with volume
                long_condition = (price <= bb_lower_val) and volume_confirmed
                short_condition = (price >= bb_upper_val) and volume_confirmed
                if long_condition:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                elif short_condition:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
        
        elif position == 1:
            # Long position - check exits
            # Stoploss (2.0x ATR)
            if price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Regime change exit
            elif is_bear:  # Flip to bear regime
                signals[i] = 0.0
                position = 0
            # Take profit at opposite BB or Alligator extreme
            elif price >= bb_upper_val:
                signals[i] = 0.0
                position = 0
            elif is_bull and price >= lips_val * 1.02:  # Extended bull
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short position - check exits
            # Stoploss (2.0x ATR)
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Regime change exit
            elif is_bull:  # Flip to bull regime
                signals[i] = 0.0
                position = 0
            # Take profit at opposite BB or Alligator extreme
            elif price <= bb_lower_val:
                signals[i] = 0.0
                position = 0
            elif is_bear and price <= jaw_val * 0.98:  # Extended bear
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Williams_Alligator_Regime_Adaptive_v1"
timeframe = "1d"
leverage = 1.0