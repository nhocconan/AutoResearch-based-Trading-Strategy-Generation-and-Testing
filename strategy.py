#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla H3/L3 breakout with 1d volume spike filter and 1w ADX > 20 regime filter
# Camarilla levels provide precise intraday support/resistance; breakouts with volume confirmation
# capture institutional moves. Weekly ADX > 20 ensures we only trade when there's sufficient trend
# strength to avoid whipsaws in ranging markets. Designed for low frequency: ~12-20 trades/year.
# Works in bull/bear: volume confirms breakout legitimacy, ADX filter avoids chop, Camarilla
# levels adapt to volatility. Uses discrete sizing (0.25) to minimize fee churn.

name = "12h_Camarilla_H3L3_Breakout_1dVolume_1wADX_Regime_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1w HTF data for ADX regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 12h bar (H3, L3, H4, L4)
    # H3 = close + 1.1*(high-low)/4, L3 = close - 1.1*(high-low)/4
    # H4 = close + 1.1*(high-low)/2, L4 = close - 1.1*(high-low)/2
    # Using previous bar's OHLC to avoid look-ahead
    prev_close = np.concatenate([[np.nan], close[:-1]])
    prev_high = np.concatenate([[np.nan], high[:-1]])
    prev_low = np.concatenate([[np.nan], low[:-1]])
    
    rang = prev_high - prev_low
    H3 = prev_close + 1.1 * rang / 4
    L3 = prev_close - 1.1 * rang / 4
    H4 = prev_close + 1.1 * rang / 2
    L4 = prev_close - 1.1 * rang / 2
    
    # 1d volume confirmation: volume > 1.8 * 20-period EMA
    vol_1d = df_1d['volume'].values
    vol_ema_20 = pd.Series(vol_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike_1d = vol_1d > (1.8 * vol_ema_20)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # 1w ADX(14) for regime filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = np.abs(high_1w[1:] - low_1w[1:])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_minus = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Wilder's smoothing
    def wilders_smoothing(x, period):
        result = np.full_like(x, np.nan)
        if len(x) >= period:
            first_val = np.nansum(x[1:period+1])
            result[period] = first_val
            for i in range(period+1, len(x)):
                result[i] = result[i-1] - (result[i-1] / period) + x[i]
        return result
    
    tr_period = 14
    tr_smoothed = wilders_smoothing(tr, tr_period)
    dm_plus_smoothed = wilders_smoothing(dm_plus, tr_period)
    dm_minus_smoothed = wilders_smoothing(dm_minus, tr_period)
    
    # DI+ and DI-
    di_plus = np.where(tr_smoothed != 0, (dm_plus_smoothed / tr_smoothed) * 100, 0)
    di_minus = np.where(tr_smoothed != 0, (dm_minus_smoothed / tr_smoothed) * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 
                  np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx = wilders_smoothing(dx, tr_period)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 20  # Need Camarilla levels and ADX
    
    for i in range(start_idx, n):
        if (np.isnan(H3[i]) or np.isnan(L3[i]) or np.isnan(H4[i]) or np.isnan(L4[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_spike_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade when ADX > 20 (sufficient trend strength)
        strong_trend = adx_aligned[i] > 20
        
        if position == 0:  # Flat - look for new entries
            if strong_trend and volume_spike_1d_aligned[i]:
                # Long: Break above H3 with volume spike
                if close[i] > H3[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: Break below L3 with volume spike
                elif close[i] < L3[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # Avoid weak/choppy markets or low volume
        
        elif position == 1:  # Long position
            # Exit: price returns to L3 (reversion to mean)
            if close[i] <= L3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price returns to H3 (reversion to mean)
            if close[i] >= H3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals