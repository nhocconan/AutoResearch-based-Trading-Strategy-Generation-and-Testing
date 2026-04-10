#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot mean reversion with 1d volume confirmation and chop filter
# - Primary: 4h price touching Camarilla H3/L3 levels from 1d OHLC for mean reversion entries
# - HTF trend: 1d EMA(50) slope determines market regime (rising = long bias allowed, falling = short bias allowed)
# - HTF volume: 1d volume > 1.3x 20-period MA for institutional participation
# - Regime filter: 4h choppiness index > 61.8 ensures ranging market (mean reversion edge)
# - Long: price <= L3 + rising 1d EMA slope + volume spike + chop > 61.8
# - Short: price >= H3 + falling 1d EMA slope + volume spike + chop > 61.8
# - Exit: price crosses 1d VWAP (mean reversion complete) or chop < 38.2 (trend emerging)
# - Position sizing: 0.25 (discrete level to minimize fee churn)
# - Target: 100-180 total trades over 4 years (25-45/year) for 4h timeframe
# - Works in bull/bear: Camarilla pivots work in all regimes, chop filter ensures mean reversion edge, volume confirms participation

name = "4h_1d_camarilla_chop_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute HTF data
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 4h choppiness index (14)
    def calculate_choppiness(high, low, close, window=14):
        atr = np.zeros_like(close)
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First period TR is just high-low
        atr = pd.Series(tr).rolling(window=window, min_periods=window).mean().values
        
        highest_high = pd.Series(high).rolling(window=window, min_periods=window).max().values
        lowest_low = pd.Series(low).rolling(window=window, min_periods=window).min().values
        
        chop = np.zeros_like(close)
        for i in range(len(close)):
            if not np.isnan(atr[i]) and atr[i] > 0 and not np.isnan(highest_high[i]) and not np.isnan(lowest_low[i]):
                chop[i] = 100 * np.log10(atr[i] * np.sqrt(window) / (highest_high[i] - lowest_low[i])) / np.log10(window)
            else:
                chop[i] = np.nan
        return chop
    
    chop = calculate_choppiness(high, low, close)
    
    # Calculate 1d Camarilla levels (based on previous day OHLC)
    camarilla_h3 = np.full_like(close_1d, np.nan)
    camarilla_l3 = np.full_like(close_1d, np.nan)
    camarilla_h3 = close_1d + 1.1 * (high_1d - low_1d) / 6
    camarilla_l3 = close_1d - 1.1 * (high_1d - low_1d) / 6
    
    # Align Camarilla levels to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Calculate 1d EMA(50) and its slope
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate EMA slope (rate of change over 3 periods)
    ema_slope = np.zeros_like(ema_1d_aligned)
    for i in range(3, len(ema_1d_aligned)):
        if not np.isnan(ema_1d_aligned[i]) and not np.isnan(ema_1d_aligned[i-3]):
            ema_slope[i] = (ema_1d_aligned[i] - ema_1d_aligned[i-3]) / 3
        else:
            ema_slope[i] = np.nan
    
    # Calculate 1d volume MA(20)
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    # Calculate 1d VWAP (typical price * volume) / cumulative volume
    typical_price_1d = (high_1d + low_1d + close_1d) / 3
    vwap_1d = (pd.Series(typical_price_1d * volume_1d).cumsum() / pd.Series(volume_1d).cumsum()).values
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    for i in range(60, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(chop[i]) or np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(ema_1d_aligned[i]) or np.isnan(ema_slope[i]) or np.isnan(volume_ma_20_1d_aligned[i]) or
            np.isnan(vwap_1d_aligned[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.3x 20-period MA
        volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        volume_confirm = volume_1d_aligned[i] > 1.3 * volume_ma_20_1d_aligned[i]
        
        # Chop regime filter: > 61.8 = ranging (good for mean reversion)
        chop_regime = chop[i] > 61.8
        # Trend emerging filter: < 38.2 = trending (exit mean reversion positions)
        trend_emerging = chop[i] < 38.2
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price <= L3 + rising 1d EMA slope + volume spike + chop > 61.8
            if (close[i] <= camarilla_l3_aligned[i] and ema_slope[i] > 0 and volume_confirm and chop_regime):
                position = 1
                signals[i] = 0.25
            # Short entry: price >= H3 + falling 1d EMA slope + volume spike + chop > 61.8
            elif (close[i] >= camarilla_h3_aligned[i] and ema_slope[i] < 0 and volume_confirm and chop_regime):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: price crosses 1d VWAP (mean reversion complete) OR chop < 38.2 (trend emerging)
            if position == 1:  # Long position
                if close[i] >= vwap_1d_aligned[i] or trend_emerging:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if close[i] <= vwap_1d_aligned[i] or trend_emerging:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals