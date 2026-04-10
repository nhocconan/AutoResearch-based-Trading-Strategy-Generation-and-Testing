#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator + Elder Ray + volume confirmation
# - Williams Alligator: Jaw (13-period SMMA smoothed 8), Teeth (8-period SMMA smoothed 5), Lips (5-period SMMA smoothed 3)
# - Bullish when Lips > Teeth > Jaw (alligator eating with mouth up)
# - Bearish when Jaw > Teeth > Lips (alligator eating with mouth down)
# - Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
# - Long when Bull Power > 0 AND alligator bullish AND volume > 1.5x 20-period volume SMA
# - Short when Bear Power < 0 AND alligator bearish AND volume > 1.5x 20-period volume SMA
# - Exit: Opposite Elder Ray signal OR ATR trailing stop (2.0x ATR)
# - Uses 1d for Alligator/Elder Ray signals and HTF 1w for regime filter (ADX > 25 = trending)
# - Position sizing: 0.25 discrete level to control drawdown and minimize fee churn
# - Target: 20-40 trades/year (80-160 total over 4 years) to minimize fee drag

name = "1d_alligator_elder_ray_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop (this IS the primary timeframe)
    df_1d = prices.copy()  # Since timeframe is 1d, prices is already 1d data
    
    # Load HTF 1w data ONCE before loop for regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return signals
    
    # Calculate SMMA (Smoothed Moving Average) for Williams Alligator
    def smma(source, length):
        """Calculate Smoothed Moving Average"""
        result = np.full_like(source, np.nan, dtype=float)
        if len(source) < length:
            return result
        # First value is simple SMA
        result[length-1] = np.mean(source[:length])
        # Subsequent values: SMMA = (PREV_SMMA * (length-1) + CLOSE) / length
        for i in range(length, len(source)):
            result[i] = (result[i-1] * (length-1) + source[i]) / length
        return result
    
    # Williams Alligator components
    jaw = smma(close, 13)  # Jaw: 13-period SMMA smoothed 8
    jaw = smma(jaw, 8)
    teeth = smma(close, 8)   # Teeth: 8-period SMMA smoothed 5
    teeth = smma(teeth, 5)
    lips = smma(close, 5)    # Lips: 5-period SMMA smoothed 3
    lips = smma(lips, 3)
    
    # Elder Ray components
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Volume confirmation
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for trailing stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 1w ADX for regime filter (only trade in trending markets: ADX > 25)
    if len(df_1w) >= 30:
        # Calculate 1w ADX
        high_1w = df_1w['high'].values
        low_1w = df_1w['low'].values
        close_1w = df_1w['close'].values
        
        # True Range
        tr1_1w = high_1w - low_1w
        tr2_1w = np.abs(high_1w - np.roll(close_1w, 1))
        tr3_1w = np.abs(low_1w - np.roll(close_1w, 1))
        tr_1w = np.maximum(tr1_1w, np.maximum(tr2_1w, tr3_1w))
        atr_1w = pd.Series(tr_1w).rolling(window=14, min_periods=14).mean().values
        
        # Directional Movement
        up_move = high_1w - np.roll(high_1w, 1)
        down_move = np.roll(low_1w, 1) - low_1w
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        # Smoothed DM and TR
        plus_di_1w = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_1w
        minus_di_1w = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_1w
        
        # ADX
        dx_1w = 100 * np.abs(plus_di_1w - minus_di_1w) / (plus_di_1w + minus_di_1w + 1e-10)
        adx_1w = pd.Series(dx_1w).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
        
        # Align 1w ADX to 1d timeframe
        adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
        trending_filter = adx_1w_aligned > 25
    else:
        trending_filter = np.ones(n, dtype=bool)  # No filter if insufficient data
    
    # Track highest high since entry for trailing stop (long)
    # Track lowest low since entry for trailing stop (short)
    highest_since_entry = np.full(n, np.nan)
    lowest_since_entry = np.full(n, np.nan)
    
    for i in range(30, n):  # Start from 30 to have sufficient lookback
        # Skip if any required data is invalid
        if (np.isnan(atr[i]) or np.isnan(volume_sma_20[i]) or
            np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(adx_1w_aligned[i]) if len(df_1w) >= 30 else False):
            signals[i] = 0.0
            continue
        
        # Williams Alligator signals
        alligator_bullish = lips[i] > teeth[i] and teeth[i] > jaw[i]
        alligator_bearish = jaw[i] > teeth[i] and teeth[i] > lips[i]
        
        # Elder Ray signals
        bull_power_positive = bull_power[i] > 0
        bear_power_negative = bear_power[i] < 0
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * volume_sma_20[i]
        
        # All conditions must be met
        long_condition = alligator_bullish and bull_power_positive and vol_confirm and trending_filter[i]
        short_condition = alligator_bearish and bear_power_negative and vol_confirm and trending_filter[i]
        
        if position == 0:  # Flat - look for entry
            if long_condition:
                position = 1
                signals[i] = 0.25
                highest_since_entry[i] = high[i]  # Initialize trailing stop
            elif short_condition:
                position = -1
                signals[i] = -0.25
                lowest_since_entry[i] = low[i]  # Initialize trailing stop
            else:
                signals[i] = 0.0
                # Carry forward NaN values for tracking
                if i > 0:
                    highest_since_entry[i] = highest_since_entry[i-1]
                    lowest_since_entry[i] = lowest_since_entry[i-1]
        elif position == 1:  # Long position - look for exit
            # Update highest high since entry
            highest_since_entry[i] = max(highest_since_entry[i-1], high[i])
            
            # ATR trailing stop: exit if price drops 2.0*ATR below highest high since entry
            trailing_stop = highest_since_entry[i] - 2.0 * atr[i]
            
            # Exit conditions: trailing stop hit OR Elder Ray turns bearish (Bull Power < 0)
            exit_condition = (close[i] < trailing_stop) or (bull_power[i] < 0)
            
            if exit_condition:
                position = 0
                signals[i] = 0.0
                # Reset tracking arrays
                highest_since_entry[i] = np.nan
                lowest_since_entry[i] = np.nan
            else:
                signals[i] = 0.25
                # Propagate tracking values
                highest_since_entry[i] = highest_since_entry[i]
                lowest_since_entry[i] = lowest_since_entry[i-1]
        else:  # position == -1 (Short position) - look for exit
            # Update lowest low since entry
            lowest_since_entry[i] = min(lowest_since_entry[i-1], low[i])
            
            # ATR trailing stop: exit if price rises 2.0*ATR above lowest low since entry
            trailing_stop = lowest_since_entry[i] + 2.0 * atr[i]
            
            # Exit conditions: trailing stop hit OR Elder Ray turns bullish (Bear Power > 0)
            exit_condition = (close[i] > trailing_stop) or (bear_power[i] > 0)
            
            if exit_condition:
                position = 0
                signals[i] = 0.0
                # Reset tracking arrays
                highest_since_entry[i] = np.nan
                lowest_since_entry[i] = np.nan
            else:
                signals[i] = -0.25
                # Propagate tracking values
                highest_since_entry[i] = highest_since_entry[i-1]
                lowest_since_entry[i] = lowest_since_entry[i]
    
    return signals