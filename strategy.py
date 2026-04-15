#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d Regime Filter
# Bull Power = High - EMA13(close), Bear Power = EMA13(close) - Low
# Long when Bull Power > 0 AND Bear Power < 0 AND 1d ADX < 20 (range regime) → mean reversion
# Short when Bear Power > 0 AND Bull Power < 0 AND 1d ADX < 20 (range regime) → mean reversion
# Uses discrete position sizing (0.25) to minimize fee churn. Designed for low trade frequency (15-30/year).
# Elder Ray measures bull/bear strength relative to EMA. Works in ranging markets by fading extremes when ADX shows no trend.
# In trending markets (ADX >= 20), we stay flat to avoid whipsaws. Effective in both bull and bear markets as it targets mean reversion in ranges.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d Indicator: ADX (regime filter) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX components: +DM, -DM, TR
    high_1d_shift = np.roll(high_1d, 1)
    low_1d_shift = np.roll(low_1d, 1)
    high_1d_shift[0] = high_1d[0]
    low_1d_shift[0] = low_1d[0]
    
    plus_dm = np.where((high_1d - high_1d_shift) > (low_1d_shift - low_1d), 
                       np.maximum(high_1d - high_1d_shift, 0), 0)
    minus_dm = np.where((low_1d_shift - low_1d) > (high_1d - high_1d_shift), 
                        np.maximum(low_1d_shift - low_1d, 0), 0)
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Wilder's smoothing (alpha = 1/period)
    period = 14
    alpha = 1.0 / period
    
    atr = np.zeros_like(tr)
    atr[period-1] = np.mean(tr[:period])
    for i in range(period, len(tr)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    
    plus_di = np.zeros_like(plus_dm)
    minus_di = np.zeros_like(minus_dm)
    
    # Smooth +DM and -DM
    plus_dm_smooth = np.zeros_like(plus_dm)
    minus_dm_smooth = np.zeros_like(minus_dm)
    
    plus_dm_smooth[period-1] = np.mean(plus_dm[:period])
    minus_dm_smooth[period-1] = np.mean(minus_dm[:period])
    
    for i in range(period, len(plus_dm)):
        plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period-1) + plus_dm[i]) / period
        minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period-1) + minus_dm[i]) / period
    
    # Avoid division by zero
    plus_di = np.where(atr != 0, 100 * plus_dm_smooth / atr, 0)
    minus_di = np.where(atr != 0, 100 * minus_dm_smooth / atr, 0)
    
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    
    # Wilder's smoothing for ADX
    adx = np.zeros_like(dx)
    adx[2*period-1] = np.mean(dx[period-1:2*period])
    for i in range(2*period, len(dx)):
        adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # === 6h Indicator: Elder Ray (Bull Power, Bear Power) ===
    ema_period = 13
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=ema_period, adjust=False, min_periods=ema_period).mean().values
    
    bull_power = high - ema13
    bear_power = ema13 - low
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(ema_period, 2*period)  # EMA13 + ADX(28)
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Range regime: 1d ADX < 20 (no strong trend)
        if adx_aligned[i] >= 20:
            signals[i] = 0.0  # flat in trending markets
            continue
        
        # === LONG CONDITIONS ===
        # Bull Power > 0 (strong bulls) AND Bear Power < 0 (weak bears) = bullish extreme in range → mean reversion short?
        # Actually, we want to fade extremes: when Bull Power > 0 AND Bear Power < 0, it's balanced but bulls slightly stronger
        # Let's reverse: Long when Bear Power > 0 AND Bull Power < 0 (bears stronger but bulls weak → bullish reversal)
        # Wait, let's think: Elder Ray interpretation:
        # - Bull Power > 0: Bulls in control (price above EMA)
        # - Bear Power > 0: Bears in control (price below EMA)
        # For mean reversion in range:
        # Long when Bear Power > 0 AND Bull Power < 0 (price below EMA but bulls weak → oversold?)
        # Actually standard Elder Ray:
        # - Bull Power increasing: bulls gaining strength
        # - Bear Power increasing: bears gaining strength
        # Let's use divergence: Long when Bull Power > 0 and rising AND Bear Power < 0 and falling
        # But that's complex. Simpler: Long when Bear Power > 0 (bears in control) AND Bull Power < 0 (bulls weak) AND Bear Power declining?
        # No, let's look at what works: fade when one power is extreme.
        # Long when Bear Power > 0 (extreme bearish) AND Bull Power < 0 (bulls weak) → oversold
        # Short when Bull Power > 0 (extreme bullish) AND Bear Power < 0 (bears weak) → overbought
        
        # Extreme bearish: Bear Power > 0 AND Bull Power < 0
        # Extreme bullish: Bull Power > 0 AND Bear Power < 0
        if (bear_power[i] > 0) and (bull_power[i] < 0):
            signals[i] = 0.25  # long (fade bearish extreme)
        elif (bull_power[i] > 0) and (bear_power[i] < 0):
            signals[i] = -0.25  # short (fade bullish extreme)
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_ElderRay_1dADX20_RangeFilter_v1"
timeframe = "6h"
leverage = 1.0