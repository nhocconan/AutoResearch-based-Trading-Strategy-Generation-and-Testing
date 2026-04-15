#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1w Bollinger Bands + ATR regime filter + volume confirmation
# In low volatility regimes (BB width at 1y low), trade breakouts from BB(20,2) with volume spike.
# In high volatility regimes, fade BB extremes with volume confirmation.
# Uses 1w timeframe for regime/structure to avoid whipsaw, 12h for execution.
# Designed for low trade frequency (15-25/year) to minimize fee drag while adapting to volatility regimes.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours to avoid datetime operations in loop
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Get 1w HTF data once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === 1w Indicators: Bollinger Bands + ATR for regime ===
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Bollinger Bands (20, 2)
    sma_20_1w = pd.Series(close_1w).rolling(window=20, min_periods=20).mean().values
    std_20_1w = pd.Series(close_1w).rolling(window=20, min_periods=20).std().values
    upper_bb_1w = sma_20_1w + (2 * std_20_1w)
    lower_bb_1w = sma_20_1w - (2 * std_20_1w)
    bb_width_1w = (upper_bb_1w - lower_bb_1w) / sma_20_1w  # normalized width
    
    # ATR(14) for volatility regime
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr_14_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align to 12h timeframe
    upper_bb_aligned = align_htf_to_ltf(prices, df_1w, upper_bb_1w)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1w, lower_bb_1w)
    bb_width_aligned = align_htf_to_ltf(prices, df_1w, bb_width_1w)
    atr_14_aligned = align_htf_to_ltf(prices, df_1w, atr_14_1w)
    
    # === 12h Indicators: Volume SMA ===
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Session filter: 08-20 UTC only
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
            
        # Volume filter: current volume > 2.0x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 2.0)
        
        # Skip if any required data is NaN
        if (np.isnan(upper_bb_aligned[i]) or np.isnan(lower_bb_aligned[i]) or
            np.isnan(bb_width_aligned[i]) or np.isnan(atr_14_aligned[i])):
            signals[i] = 0.0
            continue
        
        # === REGIME DETECTION BASED ON VOLATILITY ===
        # Low volatility regime: BB width at 1-year low (52-week lookback)
        # High volatility regime: BB width above 1-year high
        # Normal regime: between
        
        # For simplicity, use percentile of BB width over last 52 weeks
        # Since we only have aligned data, approximate with recent lookback
        # Use 26-period (6 months) lookback for BB width percentile
        if i >= 26:
            bb_width_slice = bb_width_aligned[i-26:i+1]
            bb_width_percentile = (bb_width_slice <= bb_width_aligned[i]).mean() * 100
        else:
            bb_width_percentile = 50  # neutral
        
        low_vol_regime = bb_width_percentile <= 20  # bottom 20%
        high_vol_regime = bb_width_percentile >= 80  # top 20%
        normal_regime = (bb_width_percentile > 20) and (bb_width_percentile < 80)
        
        # === ENTRY LOGIC ===
        if vol_confirm:
            # Low volatility regime: breakout strategy
            if low_vol_regime:
                # Long breakout above upper BB
                if close[i] > upper_bb_aligned[i]:
                    signals[i] = 0.25
                # Short breakdown below lower BB
                elif close[i] < lower_bb_aligned[i]:
                    signals[i] = -0.25
            
            # High volatility regime: mean reversion at BB extremes
            elif high_vol_regime:
                # Long near lower BB (oversold)
                if close[i] < lower_bb_aligned[i] * 1.01:  # within 1% of lower BB
                    signals[i] = 0.20
                # Short near upper BB (overbought)
                elif close[i] > upper_bb_aligned[i] * 0.99:  # within 1% of upper BB
                    signals[i] = -0.20
            
            # Normal regime: no clear edge, stay flat
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_BB_Width_ATR_Regime_VolFilter_v1"
timeframe = "12h"
leverage = 1.0