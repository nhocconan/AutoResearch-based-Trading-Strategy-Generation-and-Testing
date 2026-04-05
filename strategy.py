#!/usr/bin/env python3
"""
Experiment #9694: 1h VWAP Reversion with 4h Trend and Volume Confirmation.
Hypothesis: Price reverts to VWAP in ranging markets (ADX < 20) and continues with 4h trend (ADX > 25) on volume spikes.
Works in bull/bear via regime filter. Targets 60-150 trades over 4 years (15-37/year) for 1h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9694_1h_vwap_reversion_4h_trend_volume_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
VWAP_PERIOD = 24  # 24 hours for VWAP calculation
VOLUME_SPIKE_MULTIPLIER = 2.0
ADX_PERIOD = 14
ADX_RANGE_THRESHOLD = 20   # ADX < 20: ranging (mean revert)
ADX_TREND_THRESHOLD = 25   # ADX > 25: trending (follow 4h)
SIGNAL_SIZE = 0.20
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_vwap(high, low, close, volume, period):
    """Calculate VWAP (Volume Weighted Average Price)"""
    typical_price = (high + low + close) / 3.0
    vwap_num = (typical_price * volume).rolling(window=period, min_periods=period).sum()
    vwap_den = volume.rolling(window=period, min_periods=period).sum()
    vwap = vwap_num / vwap_den
    return vwap.fillna(method='ffill').values  # forward fill for initial period

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]  # first TR is just high-low
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_adx(high, low, close, period):
    """Calculate ADX using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    
    plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values / atr
    
    dx = np.where((plus_di + minus_di) != 0, 
                  100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (4h for trend direction)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h EMA for trend direction
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Calculate LTF indicators (1h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # VWAP for mean reversion target
    vwap = calculate_vwap(high, low, close, volume, VWAP_PERIOD)
    
    # Volume moving average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ADX for regime filtering
    adx = calculate_adx(high, low, close, ADX_PERIOD)
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(VWAP_PERIOD, 20, ADX_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(ema_4h_aligned[i]) or np.isnan(vwap[i]) or np.isnan(volume_ma[i]) or np.isnan(adx[i]) or np.isnan(atr[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Volume spike confirmation
        volume_spike = volume[i] > (volume_ma[i] * VOLUME_SPIKE_MULTIPLIER)
        
        # Regime filters
        ranging = adx[i] < ADX_RANGE_THRESHOLD   # ADX < 20: ranging market
        trending = adx[i] > ADX_TREND_THRESHOLD  # ADX > 25: trending market
        
        # Mean reversion in ranging markets: fade deviation from VWAP
        mean_rev_long = ranging and volume_spike and close[i] < vwap[i]
        mean_rev_short = ranging and volume_spike and close[i] > vwap[i]
        
        # Trend following in trending markets: follow 4h EMA
        trend_follow_long = trending and volume_spike and close[i] > ema_4h_aligned[i]
        trend_follow_short = trending and volume_spike and close[i] < ema_4h_aligned[i]
        
        # Entry conditions
        long_entry = mean_rev_long or trend_follow_long
        short_entry = mean_rev_short or trend_follow_short
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals