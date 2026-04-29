#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume spike confirmation and choppiness regime filter
# Long when price breaks above 20-period Donchian high AND volume > 2.0x 20-period average AND Chop > 61.8 (ranging market mean reversion)
# Short when price breaks below 20-period Donchian low AND volume > 2.0x 20-period average AND Chop > 61.8
# Uses ATR trailing stop (2.5x ATR) for risk management
# Discrete position sizing (0.25) to minimize fee churn
# Target: 30-60 trades/year on 4h timeframe to avoid fee drag while capturing strong reversals in ranging markets
# Choppiness filter ensures we only trade mean reversion in choppy/range-bound conditions (avoids trending whipsaw)
# Volume confirmation ensures breakouts have strong participation
# Works in bull markets via long reversals from Donchian high in chop
# Works in bear markets via short reversals from Donchian low in chop

name = "4h_DonchianBreakout_VolumeSpike_ChopFilter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR for stoploss (using 14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume spike confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma_20
    
    # Calculate Choppiness Index (14-period) for regime filter
    # Chop = 100 * log10(sum(ATR(14)) / log10(highest_high - lowest_low)) / log10(14)
    # We'll use a simplified version: Chop = 100 * log10(sum(tr) / log10(hhh - lll)) / log10(14)
    # But for efficiency, we'll use: Chop = 100 * log10(sum(tr) / log10(highest_high - lowest_low)) / log10(14)
    # Actually, standard formula: CHOP = 100 * LOG10(SUM(ATR1, n) / (LOG10(HHV - LLV) * LOG10(n))) / LOG10(n)
    # Simplified: we'll calculate true range sum and price range
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    hh_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    price_range = hh_14 - ll_14
    # Avoid division by zero and log of zero
    chop = np.where(
        (price_range > 0) & (atr_14 > 0),
        100 * np.log10(atr_14 / price_range) / np.log10(14),
        50.0  # neutral when undefined
    )
    
    # Choppiness regime: Chop > 61.8 indicates ranging market (good for mean reversion)
    chop_regime = chop > 61.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    start_idx = 20  # warmup for Donchian and volume MA
    
    for i in range(start_idx, n):
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_atr = atr[i]
        curr_donchian_high = donchian_high[i]
        curr_donchian_low = donchian_low[i]
        curr_vol_spike = vol_spike[i]
        curr_chop_regime = chop_regime[i]
        
        # Handle exits and stoploss
        if position == 1:  # Long position
            # Update highest high since entry
            highest_high_since_entry = max(highest_high_since_entry, curr_high)
            # Trailing stop: 2.5 * ATR below highest high
            stop_price = highest_high_since_entry - 2.5 * curr_atr
            # Exit conditions: price below trailing stop
            if curr_close < stop_price:
                signals[i] = 0.0
                position = 0
                highest_high_since_entry = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_low_since_entry = min(lowest_low_since_entry, curr_low)
            # Trailing stop: 2.5 * ATR above lowest low
            stop_price = lowest_low_since_entry + 2.5 * curr_atr
            # Exit conditions: price above trailing stop
            if curr_close > stop_price:
                signals[i] = 0.0
                position = 0
                lowest_low_since_entry = 0.0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above Donchian high AND volume spike AND chop regime (ranging)
            if curr_close > curr_donchian_high and curr_vol_spike and curr_chop_regime:
                signals[i] = 0.25
                position = 1
                highest_high_since_entry = curr_high
            # Short entry: price breaks below Donchian low AND volume spike AND chop regime (ranging)
            elif curr_close < curr_donchian_low and curr_vol_spike and curr_chop_regime:
                signals[i] = -0.25
                position = -1
                lowest_low_since_entry = curr_low
            else:
                signals[i] = 0.0
    
    return signals