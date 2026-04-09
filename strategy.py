#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 12h ADX trend strength filter + volume spike confirmation
# - Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13 (measures buying/selling pressure)
# - Primary signal: Bull Power > 0 AND Bear Power < 0 (both indicate strong directional momentum)
# - Trend filter: 12h ADX(14) > 25 ensures we only trade in strong trends (avoids chop)
# - Volume confirmation: 6h volume > 1.5 x 20-period EMA volume (avoids low-participation false signals)
# - Position size: 0.25 discrete level to minimize fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) per 6h strategy guidelines
# - Works in bull/bear: ADX filter ensures we only trade when trend is strong, Elder Ray confirms direction,
#   volume spike confirms institutional participation. Avoids whipsaws in ranging markets.

name = "6h_12h_elderray_adx_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Pre-compute 12h indicators
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # 12h ADX(14) for trend strength
    # ADX calculation requires +DM, -DM, and TR
    up_move = np.diff(high_12h, prepend=high_12h[0])
    down_move = np.diff(low_12h, prepend=low_12h[0]) * -1  # invert to positive
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    tr1 = np.abs(np.diff(high_12h, prepend=high_12h[0]))
    tr2 = np.abs(np.diff(low_12h, prepend=low_12h[0]))
    tr3 = np.abs(np.diff(close_12h, prepend=close_12h[0]))
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # first value is simple average
        result[period-1] = np.mean(data[:period])
        # subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    period = 14
    atr_12h = wilders_smoothing(true_range, period)
    plus_di_12h = 100 * wilders_smoothing(plus_dm, period) / atr_12h
    minus_di_12h = 100 * wilders_smoothing(minus_dm, period) / atr_12h
    dx_12h = 100 * np.abs(plus_di_12h - minus_di_12h) / (plus_di_12h + minus_di_12h)
    adx_12h = wilders_smoothing(dx_12h, period)
    
    # Align 12h ADX to 6h timeframe (completed 12h bar only)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # 6h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 6h EMA13 for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # Buying power: ability to push price above EMA
    bear_power = low - ema_13   # Selling power: ability to push price below EMA
    
    # 6h volume confirmation: volume > 1.5 x 20-period EMA volume
    volume_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(adx_12h_aligned[i]) or
            np.isnan(bull_power[i]) or
            np.isnan(bear_power[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Elder Ray weakening OR trend weakening
            if bull_power[i] <= 0 or adx_12h_aligned[i] < 20:  # long exits when buying power fades or trend weakens
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Elder Ray weakening OR trend weakening
            if bear_power[i] >= 0 or adx_12h_aligned[i] < 20:  # short exits when selling power fades or trend weakens
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for strong directional momentum with trend and volume confirmation
            # Long: Bull Power > 0 (buying pressure) AND Bear Power < 0 (no selling pressure) AND strong trend AND volume spike
            if bull_power[i] > 0 and bear_power[i] < 0 and adx_12h_aligned[i] > 25 and volume_spike[i]:
                position = 1
                signals[i] = 0.25
            # Short: Bear Power < 0 (selling pressure) AND Bull Power > 0 (no buying pressure) AND strong trend AND volume spike
            elif bear_power[i] < 0 and bull_power[i] > 0 and adx_12h_aligned[i] > 25 and volume_spike[i]:
                position = -1
                signals[i] = -0.25
    
    return signals