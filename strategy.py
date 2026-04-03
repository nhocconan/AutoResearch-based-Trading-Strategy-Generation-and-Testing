#!/usr/bin/env python3
"""
Experiment #194: 1h Mean Reversion with 4h/1d Trend Filter and Volume Confirmation

HYPOTHESIS: In ranging markets (common in 2025 BTC/ETH), price reverts to the 1h VWAP 
when aligned with higher timeframe trend (4h/1d) and confirmed by volume spikes. 
Uses discrete position sizing (0.20) and session filter (08-20 UTC) to minimize fee drag. 
Targets 15-37 trades/year (60-150 total over 4 years) by requiring confluence of: 
1) Price deviation from 1h VWAP (>1.5 ATR), 
2) 4h/1d trend alignment (EMA21), 
3) Volume spike (>1.8x average), 
4) Active trading session. 
ATR-based stoploss (2.0) manages risk. Works in bull (buy dips in uptrend) and 
sell rallies in downtrend) and bear (fade rallies in downtrend, buy dips in uptrend) 
markets by following higher timeframe direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_194_1h_vwap_mean_reversion_4h_1d_trend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute session hours (08-20 UTC) - avoid datetime operations in loop
    hours = prices.index.hour  # prices.index is DatetimeIndex from parquet
    
    # === HTF: 4h and 1d EMA21 for trend alignment (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate EMA(21) on HTF data
    ema_4h = pd.Series(df_4h['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
    
    # Align HTF EMA to 1h timeframe with shift(1) for completed bars only
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === 1h Indicators: VWAP and ATR(14) ===
    # Typical price for VWAP
    typical_price = (high + low + close) / 3.0
    # VWAP = cumulative(TP * volume) / cumulative(volume)
    cum_vol = np.cumsum(volume)
    cum_vol_tp = np.cumsum(typical_price * volume)
    vwap = np.divide(cum_vol_tp, cum_vol, out=np.full_like(cum_vol_tp, np.nan), where=cum_vol!=0)
    
    # ATR(14) for stoploss and deviation threshold
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 1h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.divide(volume, vol_ma_20, out=np.ones_like(volume), where=vol_ma_20!=0)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # Position sizing (20% of capital) - discrete level to minimize churn
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 100  # Sufficient for VWAP, ATR, EMA alignment
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(vwap[i]) or np.isnan(atr_14[i]) or 
            np.isnan(ema_4h_aligned[i]) or np.isnan(ema_1d_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC only
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Calculate deviation from VWAP in ATR units
        deviation = close[i] - vwap[i]
        deviation_atr = abs(deviation) / atr_14[i] if atr_14[i] > 0 else 0
        
        # Volume confirmation: require spike > 1.8x average
        volume_spike = vol_ratio[i] > 1.8
        
        # Trend alignment: price relative to HTF EMA21
        price_above_4h_ema = close[i] > ema_4h_aligned[i]
        price_below_4h_ema = close[i] < ema_4h_aligned[i]
        price_above_1d_ema = close[i] > ema_1d_aligned[i]
        price_below_1d_ema = close[i] < ema_1d_aligned[i]
        
        # Require both 4h and 1d trend to agree (stronger filter)
        bullish_trend = price_above_4h_ema and price_above_1d_ema
        bearish_trend = price_below_4h_ema and price_below_1d_ema
        
        # Mean reentry conditions:
        # Long: price significantly below VWAP (>1.5 ATR) + bullish HTF trend + volume spike
        long_condition = (deviation < -1.5 * atr_14[i]) and bullish_trend and volume_spike
        
        # Short: price significantly above VWAP (>1.5 ATR) + bearish HTF trend + volume spike
        short_condition = (deviation > 1.5 * atr_14[i]) and bearish_trend and volume_spike
        
        if in_position:
            bars_since_entry += 1
            
            # ATR-based stoploss (2.0 ATR)
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.0 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Take profit: return to VWAP (mean reversion complete)
                if close[i] >= vwap[i]:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.0 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Take profit: return to VWAP
                if close[i] <= vwap[i]:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Minimum holding period of 2 bars to reduce churn
            if bars_since_entry < 2:
                signals[i] = position_side * SIZE
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # New position entry
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals