#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d volume confirmation and ADX trend filter.
# Long when price breaks above Donchian upper band AND 1d volume > 1.5x 20-period average AND ADX(14) > 25.
# Short when price breaks below Donchian lower band AND 1d volume > 1.5x 20-period average AND ADX(14) > 25.
# Exit when price returns to Donchian middle band (mean reversion) or opposite breakout occurs.
# Uses discrete position size 0.25. Designed to capture strong trending moves with volume confirmation
# while avoiding choppy markets via ADX filter. Target: 80-160 total trades over 4 years (20-40/year).

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h Indicators: Donchian(20) and ADX(14) ===
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = highest_high_20
    donchian_lower = lowest_low_20
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    # ADX calculation
    plus_dm = pd.Series(high).diff()
    minus_dm = pd.Series(low).diff().abs()
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
    
    tr1 = pd.Series(high).diff()
    tr2 = pd.Series(low).diff().abs()
    tr3 = pd.Series(close).shift(1).diff().abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # === 1d Indicators: Volume Spike ===
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_spike = volume > (1.5 * vol_ma_1d_aligned)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 60
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(donchian_middle[i]) or
            np.isnan(adx[i]) or np.isnan(volume_spike[i]) or not session_filter[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_spike = volume_spike[i]
        adx_val = adx[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price returns to middle band (mean reversion)
            if price <= donchian_middle[i]:
                exit_signal = True
            # Exit if opposite breakout occurs
            elif price < donchian_lower[i]:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price returns to middle band (mean reversion)
            if price >= donchian_middle[i]:
                exit_signal = True
            # Exit if opposite breakout occurs
            elif price > donchian_upper[i]:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Breakout above upper band + volume spike + ADX > 25
            if (price > donchian_upper[i] and vol_spike and adx_val > 25):
                signals[i] = 0.25
                position = 1
            
            # SHORT: Breakout below lower band + volume spike + ADX > 25
            elif (price < donchian_lower[i] and vol_spike and adx_val > 25):
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_Donchian20_1dVolumeSpike_ADXFilter_V1"
timeframe = "6h"
leverage = 1.0