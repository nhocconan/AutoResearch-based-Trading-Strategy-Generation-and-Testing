#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with volume confirmation and 4h/1d regime filter
# - Long: price breaks above H3 Camarilla pivot (4h), volume > 1.3x 20-period avg, 4h close > 1d VWAP (bull regime)
# - Short: price breaks below L3 Camarilla pivot (4h), volume > 1.3x 20-period avg, 4h close < 1d VWAP (bear regime)
# - Exit: price returns to Camarilla pivot point (PP) or opposite H4/L4 level
# - Session filter: 08-20 UTC to avoid low-volume Asian session noise
# - Target: 15-25 trades/year (60-100 total over 4 years) to stay within fee drag limits
# - Works in both bull/bear by using 4h/1d regime filters to align with higher timeframe trend

name = "1h_4h_1d_camarilla_pivot_volume_v1"
timeframe = "1h"
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
    
    # Pre-compute session filter (08-20 UTC) ONCE before loop
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop for Camarilla pivots and trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return signals
    
    # Calculate 4h Camarilla pivots (based on previous 4h bar's OHLC)
    # Camarilla levels: PP = (H+L+C)/3, Range = H-L
    # H4 = PP + Range * 1.1/2, L4 = PP - Range * 1.1/2
    # H3 = PP + Range * 1.1/4, L3 = PP - Range * 1.1/4
    # H2 = PP + Range * 1.1/6, L2 = PP - Range * 1.1/6
    # H1 = PP + Range * 1.1/12, L1 = PP - Range * 1.1/12
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Use previous bar's OHLC to avoid look-ahead (current bar still forming)
    prev_high_4h = np.roll(high_4h, 1)
    prev_low_4h = np.roll(low_4h, 1)
    prev_close_4h = np.roll(close_4h, 1)
    prev_high_4h[0] = np.nan  # First bar has no previous
    prev_low_4h[0] = np.nan
    prev_close_4h[0] = np.nan
    
    # Calculate Camarilla levels using previous bar
    pp_4h = (prev_high_4h + prev_low_4h + prev_close_4h) / 3
    range_4h = prev_high_4h - prev_low_4h
    
    h3_4h = pp_4h + range_4h * 1.1 / 4
    l3_4h = pp_4h - range_4h * 1.1 / 4
    h4_4h = pp_4h + range_4h * 1.1 / 2
    l4_4h = pp_4h - range_4h * 1.1 / 2
    
    # Align 4h Camarilla levels to 1h timeframe (wait for completed 4h bar)
    h3_4h_aligned = align_htf_to_ltf(prices, df_4h, h3_4h)
    l3_4h_aligned = align_htf_to_ltf(prices, df_4h, l3_4h)
    h4_4h_aligned = align_htf_to_ltf(prices, df_4h, h4_4h)
    l4_4h_aligned = align_htf_to_ltf(prices, df_4h, l4_4h)
    pp_4h_aligned = align_htf_to_ltf(prices, df_4h, pp_4h)
    
    # Load 1d data ONCE before loop for VWAP trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return signals
    
    # Calculate 1d VWAP (typical price * volume) / cumulative volume
    typical_price_1d = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3
    vp_1d = typical_price_1d * df_1d['volume'].values
    cum_vp_1d = np.nancumsum(vp_1d)
    cum_vol_1d = np.nancumsum(df_1d['volume'].values)
    vwap_1d = np.divide(cum_vp_1d, cum_vol_1d, out=np.full_like(cum_vp_1d, np.nan), where=cum_vol_1d!=0)
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # Pre-compute 1h volume confirmation (20-period average)
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid or outside session
        if (not in_session[i] or
            np.isnan(h3_4h_aligned[i]) or np.isnan(l3_4h_aligned[i]) or
            np.isnan(h4_4h_aligned[i]) or np.isnan(l4_4h_aligned[i]) or
            np.isnan(pp_4h_aligned[i]) or np.isnan(vwap_1d_aligned[i]) or
            np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        
        # Volume confirmation: current volume > 1.3x 20-period average
        vol_confirm = volume_current > 1.3 * volume_sma_20[i]
        
        # Regime filters: 4h close > 1d VWAP for bull bias, < for bear bias
        # Use 4h close aligned to 1h (previous completed 4h bar's close)
        df_4h_close = df_4h['close'].values
        close_4h_aligned = align_htf_to_ltf(prices, df_4h, df_4h_close)
        bull_regime = close_4h_aligned[i] > vwap_1d_aligned[i]
        bear_regime = close_4h_aligned[i] < vwap_1d_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long breakout: price above H3 Camarilla, volume confirmation, bull regime
        if close_price > h3_4h_aligned[i] and vol_confirm and bull_regime:
            enter_long = True
        
        # Short breakout: price below L3 Camarilla, volume confirmation, bear regime
        if close_price < l3_4h_aligned[i] and vol_confirm and bear_regime:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price returns to pivot point or breaks below L4
            exit_long = close_price <= pp_4h_aligned[i] or close_price < l4_4h_aligned[i]
        elif position == -1:
            # Exit short if price returns to pivot point or breaks above H4
            exit_short = close_price >= pp_4h_aligned[i] or close_price > h4_4h_aligned[i]
        
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