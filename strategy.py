#!/usr/bin/env python3
"""
Experiment #024: 6h Williams Alligator + ADX Trend Regime

HYPOTHESIS: Williams Alligator (Jaw/Teeth/Lips) is a multi-smoothed MA system
that captures institutional trend structure. Combined with ADX for regime 
filtering (trending vs ranging), this should:
- Go long when Alligator lines spread bullish + ADX > 20 (uptrend)
- Go short when Alligator lines spread bearish + ADX > 20 (downtrend)
- Stay flat when ADX < 20 (no trend = avoid whipsaws)
- Works in both bull (buy pullbacks to Alligator) and bear (short rallies)

This is a genuinely novel combination - Williams Alligator hasn't been tested
in this session despite being a well-known indicator.

TIMEFRAME: 6h primary
HTF: 1d for trend confirmation
TARGET: 75-200 total trades over 4 years (12-37/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_alligator_adx_regime_v1"
timeframe = "6h"
leverage = 1.0

def calculate_alligator(high, low, period=13):
    """
    Williams Alligator
    - Jaw (blue): SMMA of close, period 13, shifted 8 bars
    - Teeth (red): SMMA of close, period 8, shifted 5 bars
    - Lips (green): SMMA of close, period 5, shifted 3 bars
    
    SMMA is essentially an EMA-like smoothing
    """
    n = len(high)
    if n < period + 10:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    def smma(series, period):
        """Smoothed Moving Average (similar to Wilder's smoothing)"""
        result = np.full(len(series), np.nan, dtype=np.float64)
        # First value is SMA
        sma_val = np.mean(series[:period])
        result[period - 1] = sma_val
        # Subsequent values use Wilder smoothing
        for i in range(period, len(series)):
            if not np.isnan(series[i]):
                result[i] = (result[i-1] * (period - 1) + series[i]) / period
        return result
    
    close = np.zeros(n)
    # Use (high + low) / 2 for Alligator per Williams
    typical = (high + low) / 2.0
    
    jaw = smma(typical, 13)  # Jaw line
    teeth = smma(typical, 8)  # Teeth line
    lips = smma(typical, 5)  # Lips line
    
    return jaw, teeth, lips

def calculate_adx(high, low, close, period=14):
    """ADX - Average Directional Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], 
                    abs(high[i] - close[i-1]), 
                    abs(low[i] - close[i-1]))
        
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    # Smooth with Wilder's method
    atr_smooth = np.full(n, np.nan, dtype=np.float64)
    plus_dm_smooth = np.full(n, np.nan, dtype=np.float64)
    minus_dm_smooth = np.full(n, np.nan, dtype=np.float64)
    
    atr_smooth[period] = np.sum(tr[1:period+1])
    plus_dm_smooth[period] = np.sum(plus_dm[1:period+1])
    minus_dm_smooth[period] = np.sum(minus_dm[1:period+1])
    
    for i in range(period + 1, n):
        atr_smooth[i] = atr_smooth[i-1] - atr_smooth[i-1]/period + tr[i]
        plus_dm_smooth[i] = plus_dm_smooth[i-1] - plus_dm_smooth[i-1]/period + plus_dm[i]
        minus_dm_smooth[i] = minus_dm_smooth[i-1] - minus_dm_smooth[i-1]/period + minus_dm[i]
    
    plus_di = np.zeros(n, dtype=np.float64)
    minus_di = np.zeros(n, dtype=np.float64)
    dx = np.zeros(n, dtype=np.float64)
    
    for i in range(period, n):
        if atr_smooth[i] > 0:
            plus_di[i] = 100 * plus_dm_smooth[i] / atr_smooth[i]
            minus_di[i] = 100 * minus_dm_smooth[i] / atr_smooth[i]
            
            di_sum = plus_di[i] + minus_di[i]
            if di_sum > 0:
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx = np.full(n, np.nan, dtype=np.float64)
    adx[period*2] = np.mean(dx[period:period*2])
    
    for i in range(period * 2 + 1, n):
        if not np.isnan(adx[i-1]):
            adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
    
    return adx

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], 
                    abs(high[i] - close[i-1]), 
                    abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d SMA for trend filter
    sma_1d = df_1d['close'].values
    # Use simple 20-period SMA of 1d close
    sma_1d_vals = pd.Series(sma_1d).rolling(window=20, min_periods=20).mean().values
    sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d_vals)
    
    # Calculate 6h indicators
    jaw, teeth, lips = calculate_alligator(high, low, period=13)
    adx = calculate_adx(high, low, close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Volume for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.25  # Conservative size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    
    warmup = 60
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(adx[i]) or np.isnan(atr_14[i]) or atr_14[i] <= 0:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        adx_val = adx[i]
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        atr_val = atr_14[i]
        
        # === ALLIGATOR TREND DIRECTION ===
        # Bullish: Lips > Teeth > Jaw (lines stacked up)
        bullish_alligator = lips_val > teeth_val > jaw_val
        # Bearish: Lips < Teeth < Jaw (lines stacked down)
        bearish_alligator = lips_val < teeth_val < jaw_val
        
        # === ALLIGATOR SPREAD (trend strength) ===
        # Wider spread = stronger trend
        spread = abs(lips_val - jaw_val)
        spread_pct = spread / jaw_val if jaw_val > 0 else 0
        
        # === ADX REGIME FILTER ===
        # ADX > 20 = trending, ADX > 25 = strong trend
        # ADX < 20 = ranging, avoid entries
        trending = adx_val > 20
        strong_trend = adx_val > 25
        
        # === 1d TREND FILTER ===
        price_above_1d = close[i] > sma_1d_aligned[i] if not np.isnan(sma_1d_aligned[i]) else True
        price_below_1d = close[i] < sma_1d_aligned[i] if not np.isnan(sma_1d_aligned[i]) else False
        
        # === VOLUME CONFIRMATION ===
        vol_confirm = vol_ratio[i] > 1.0  # At least average volume
        
        desired_signal = 0.0
        
        if not in_position:
            # === NEW LONG ENTRY ===
            # Alligator bullish + trending + 1d aligned
            if bullish_alligator and trending:
                # Need 1d trend aligned OR strong ADX
                if price_above_1d or strong_trend:
                    if vol_confirm:
                        desired_signal = SIZE
            
            # === NEW SHORT ENTRY ===
            # Alligator bearish + trending + 1d aligned
            if bearish_alligator and trending:
                # Need 1d trend aligned OR strong ADX
                if price_below_1d or strong_trend:
                    if vol_confirm:
                        desired_signal = -SIZE
        
        # === STOPLOSS CHECK ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            # Long: stop below entry - 2.5 ATR
            stop_price = entry_price - 2.5 * entry_atr
            if low[i] < stop_price:
                stoploss_triggered = True
            # Also exit if Alligator turns bearish
            if bearish_alligator:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            # Short: stop above entry + 2.5 ATR
            stop_price = entry_price + 2.5 * entry_atr
            if high[i] > stop_price:
                stoploss_triggered = True
            # Also exit if Alligator turns bullish
            if bullish_alligator:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_val
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
        
        signals[i] = desired_signal
    
    return signals