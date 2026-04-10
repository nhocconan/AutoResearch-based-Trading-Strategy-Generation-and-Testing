#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d regime filter and volume confirmation
# - Williams Alligator: Jaw (13-period SMMA shifted 8), Teeth (8-period SMMA shifted 5), Lips (5-period SMMA shifted 3)
# - Long when Lips > Teeth > Jaw (bullish alignment) AND price > Lips AND 1d ADX > 25 (trending regime) AND volume confirmation
# - Short when Lips < Teeth < Jaw (bearish alignment) AND price < Lips AND 1d ADX > 25 (trending regime) AND volume confirmation
# - Volume confirmation: 12h volume > 1.3x 20-period volume SMA
# - Exit: opposite Alligator alignment or loss of volume confirmation
# - Position sizing: 0.25 discrete level to minimize fee drag
# - Target: 12-37 trades/year on 12h timeframe to stay within fee drag limits
# - Uses 1d timeframe for ADX regime filter to avoid overtrading

name = "12h_1d_alligator_regime_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate Williams Alligator components (SMMA = smoothed moving average)
    def smma(data, period):
        """Smoothed Moving Average"""
        if len(data) < period:
            return np.full(len(data), np.nan)
        result = np.full(len(data), np.nan)
        # First value is simple SMA
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_PRICE) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    # Alligator components
    jaw = smma(close, 13)  # Jaw: 13-period SMMA
    teeth = smma(close, 8)  # Teeth: 8-period SMMA
    lips = smma(close, 5)   # Lips: 5-period SMMA
    
    # Shift the lines as per Alligator definition
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # Set NaN for shifted values that rolled from end
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # Calculate 1d ADX for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align with index 0
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed TR, +DM, -DM (using Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        """Wilder's smoothing (EMA with alpha=1/period)"""
        if len(data) < period:
            return np.full(len(data), np.nan)
        result = np.full(len(data), np.nan)
        # First value is simple average
        result[period-1] = np.nanmean(data[:period])
        # Subsequent values: Wilder's smoothing
        alpha = 1.0 / period
        for i in range(period, len(data)):
            if np.isnan(result[i-1]):
                result[i] = np.nanmean(data[i-period+1:i+1])
            else:
                result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    period = 14
    tr_smoothed = wilders_smoothing(tr, period)
    plus_dm_smoothed = wilders_smoothing(plus_dm, period)
    minus_dm_smoothed = wilders_smoothing(minus_dm, period)
    
    # DI+ and DI-
    plus_di = 100 * plus_dm_smoothed / tr_smoothed
    minus_di = 100 * minus_dm_smoothed / tr_smoothed
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smoothing(dx, period)
    
    # Align HTF indicators to LTF
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 12h volume SMA for regime filter
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(50, n):  # Start after warmup for indicators
        # Skip if any required data is invalid
        if (np.isnan(lips_shifted[i]) or np.isnan(teeth_shifted[i]) or np.isnan(jaw_shifted[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 12h volume > 1.3x 20-period volume SMA
        vol_confirm = volume[i] > 1.3 * volume_sma_20[i]
        
        # Alligator alignment signals
        bullish_alignment = lips_shifted[i] > teeth_shifted[i] > jaw_shifted[i]
        bearish_alignment = lips_shifted[i] < teeth_shifted[i] < jaw_shifted[i]
        
        # Price relative to Lips (entry filter)
        price_above_lips = close[i] > lips_shifted[i]
        price_below_lips = close[i] < lips_shifted[i]
        
        # Regime filter: 1d ADX > 25 (trending market)
        trending_regime = adx_aligned[i] > 25
        
        # Entry signals
        enter_long = bullish_alignment and price_above_lips and vol_confirm and trending_regime
        enter_short = bearish_alignment and price_below_lips and vol_confirm and trending_regime
        
        # Exit conditions: opposite alignment or loss of confirmation
        exit_long = bearish_alignment or not vol_confirm or not trending_regime
        exit_short = bullish_alignment or not vol_confirm or not trending_regime
        
        if position == 0:  # Flat - look for entry
            if enter_long:
                position = 1
                signals[i] = 0.25
            elif enter_short:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals