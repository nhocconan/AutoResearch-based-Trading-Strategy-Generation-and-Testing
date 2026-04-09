#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h/1d Camarilla pivot breakout + volume confirmation + chop regime filter
# Camarilla pivots provide intraday support/resistance levels derived from prior day's range
# Breakout above/below H3/L3 with volume confirmation captures institutional moves
# Chop regime filter adapts strategy: CHOP > 61.8 = range (fade extremes), CHOP < 38.2 = trend (follow breakout)
# Works in bull/bear: regime filter prevents whipsaws in ranging markets, breakout catches strong trends
# Target: 60-150 total trades over 4 years (15-37/year) with discrete sizing 0.20
# Timeframe: 1h (primary), HTF: 4h/1d for direction

name = "1h_4h_1d_camarilla_breakout_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # === 4h Indicators for trend direction ===
    close_4h = df_4h['close'].values
    # 4h EMA(21) for trend filter
    ema_4h = pd.Series(close_4h).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # === 1d Indicators for Camarilla pivots and Chop ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d average volume (20-period) for confirmation
    volume_s_1d = pd.Series(volume_1d)
    avg_volume_1d = volume_s_1d.rolling(window=20, min_periods=20).mean().values
    avg_volume_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_1d)
    
    # 1d Choppiness Index (CHOP) for regime detection
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Wilder's ATR(14)
    def wilders_smoothing(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        alpha = 1.0 / period
        result = np.full(len(values), np.nan)
        result[period-1] = np.nanmean(values[:period])
        for i in range(period, len(values)):
            result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
        return result
    
    atr_1d = wilders_smoothing(tr, 14)
    
    # Highest high and lowest low over 14 periods
    hh_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chop calculation: 100 * log10(sum(atr14) / (hh14 - ll14)) / log10(14)
    sum_atr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    range_14 = hh_1d - ll_1d
    chop_1d = np.where(range_14 != 0, 
                       100 * np.log10(sum_atr_14 / range_14) / np.log10(14), 
                       50)  # neutral when range is zero
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # 1d Camarilla pivots (based on prior day's range)
    # Camarilla levels: H4, H3, H2, H1, L1, L2, L3, L4
    # We use H3 and L3 as breakout levels
    range_1d = high_1d - low_1d
    camarilla_h3 = close_1d + range_1d * 1.1 / 4
    camarilla_l3 = close_1d - range_1d * 1.1 / 4
    
    # Align Camarilla levels to 1h timeframe (wait for 1d bar close)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid or outside session
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(avg_volume_1d_aligned[i]) or
            np.isnan(chop_1d_aligned[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1h volume > 1.2x 1d average volume
        volume_confirmed = volume[i] > 1.2 * avg_volume_1d_aligned[i]
        
        # Regime filter: CHOP < 38.2 = trending (follow breakout), CHOP > 61.8 = range (fade extremes)
        trending_regime = chop_1d_aligned[i] < 38.2
        ranging_regime = chop_1d_aligned[i] > 61.8
        
        if position == 1:  # Long position
            # Exit: price closes below Camarilla L3 OR regime shifts to ranging (fade instead of follow)
            if close[i] < camarilla_l3_aligned[i] or ranging_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price closes above Camarilla H3 OR regime shifts to ranging
            if close[i] > camarilla_h3_aligned[i] or ranging_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Entry logic based on regime
            if trending_regime and volume_confirmed:
                # Follow breakout in trending regime
                if close[i] > camarilla_h3_aligned[i] and close[i] > ema_4h_aligned[i]:
                    position = 1
                    signals[i] = 0.20
                elif close[i] < camarilla_l3_aligned[i] and close[i] < ema_4h_aligned[i]:
                    position = -1
                    signals[i] = -0.20
            elif ranging_regime and volume_confirmed:
                # Fade extremes in ranging regime (mean reversion at H3/L3)
                if close[i] < camarilla_l3_aligned[i]:
                    position = 1
                    signals[i] = 0.20
                elif close[i] > camarilla_h3_aligned[i]:
                    position = -1
                    signals[i] = -0.20
    
    return signals