#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla Pivot Breakout with 1w Volume Confirmation and ADX Regime Filter
# Uses weekly ADX to define regime: ADX>25 = trending (trade Camarilla breakouts), ADX<20 = range (fade to pivots)
# Camarilla levels calculated from prior 1d OHLC: H4/L4 = primary breakout levels
# Entry: Long when close > H4 with volume > 1.5x 20-period average in trending regime OR when close < L4 with volume confirmation in ranging regime
# Exit: Opposite signal or price returns to pivot point (PP)
# Designed for low frequency (30-100 trades over 4 years) with clear bull/bear logic

name = "1d_Camarilla_1wADX_Regime_Volume_Breakout_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w HTF data for regime filter and volume average
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # 1w ADX(14) calculation for regime detection
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
    
    # 1w 20-period volume average for confirmation
    vol_20_1w = pd.Series(df_1w['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_20_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_20_1w)
    
    # Calculate 1d Camarilla levels from prior day OHLC
    # Camarilla: PP = (H+L+C)/3, Range = H-L
    # H4 = PP + 1.1 * Range / 2, L4 = PP - 1.1 * Range / 2
    prior_high = np.concatenate([[np.nan], high[:-1]])
    prior_low = np.concatenate([[np.nan], low[:-1]])
    prior_close = np.concatenate([[np.nan], close[:-1]])
    
    camarilla_pp = (prior_high + prior_low + prior_close) / 3
    camarilla_range = prior_high - prior_low
    camarilla_h4 = camarilla_pp + 1.1 * camarilla_range / 2
    camarilla_l4 = camarilla_pp - 1.1 * camarilla_range / 2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 34  # Need ADX and volume average
    
    for i in range(start_idx, n):
        if (np.isnan(adx_aligned[i]) or np.isnan(vol_20_1w_aligned[i]) or 
            np.isnan(camarilla_h4[i]) or np.isnan(camarilla_l4[i]) or
            np.isnan(camarilla_pp[i])):
            signals[i] = 0.0
            continue
        
        # Regime filters
        trending = adx_aligned[i] > 25
        ranging = adx_aligned[i] < 20
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * vol_20_1w_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Trending regime: Camarilla breakout
            if trending and volume_confirmed:
                # Long: Close breaks above H4
                if close[i] > camarilla_h4[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: Close breaks below L4
                elif close[i] < camarilla_l4[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            # Ranging regime: Camarilla mean reversion (fade extremes)
            elif ranging and volume_confirmed:
                # Long: Close below L4 (oversold) - mean reversion long
                if close[i] < camarilla_l4[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: Close above H4 (overbought) - mean reversion short
                elif close[i] > camarilla_h4[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # No volume confirmation or transition regime
        
        elif position == 1:  # Long position
            # Exit conditions
            exit_long = False
            if trending:
                # Exit trending long when price returns to PP (mean reversion)
                if close[i] <= camarilla_pp[i]:
                    exit_long = True
            elif ranging:
                # Exit ranging long when price reaches H4 (overbought)
                if close[i] >= camarilla_h4[i]:
                    exit_long = True
            else:
                # Transition regime - exit on any adverse move
                if close[i] <= camarilla_pp[i]:
                    exit_long = True
            
            if exit_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            if trending:
                # Exit trending short when price returns to PP (mean reversion)
                if close[i] >= camarilla_pp[i]:
                    exit_short = True
            elif ranging:
                # Exit ranging short when price reaches L4 (oversold)
                if close[i] <= camarilla_l4[i]:
                    exit_short = True
            else:
                # Transition regime - exit on any adverse move
                if close[i] >= camarilla_pp[i]:
                    exit_short = True
            
            if exit_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals