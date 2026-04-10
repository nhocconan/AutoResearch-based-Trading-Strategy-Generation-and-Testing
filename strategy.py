#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout + 4h trend filter + volume confirmation
# - Primary signal: 1h price breaks above/below Camarilla pivot levels (H3/L3) from prior 4h bar
# - Trend filter: 4h close > EMA(50) for longs, < EMA(50) for shorts (institutional trend)
# - Volume filter: 1h volume > 1.3x 20-period average volume (momentum confirmation)
# - Position size: 0.20 discrete level to minimize fee churn
# - Stoploss: 1.5x ATR(14) on 1h
# - Session filter: 08-20 UTC to avoid low-volume Asian session
# - Target: 15-37 trades/year (60-150 total over 4 years) per 1h strategy guidelines
# - Works in bull/bear: Camarilla levels provide dynamic support/resistance; trend filter avoids counter-trend trades

name = "1h_4h_camarilla_breakout_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 60:
        return np.zeros(n)
    
    # Pre-compute 4h EMA(50) for trend filter
    close_4h = df_4h['close'].values
    ema_50 = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50)
    
    # Pre-compute 1h ATR(14) for stoploss
    high_1h = prices['high'].values
    low_1h = prices['low'].values
    close_1h = prices['close'].values
    
    tr1 = high_1h - low_1h
    tr2 = np.abs(high_1h - np.roll(close_1h, 1))
    tr3 = np.abs(low_1h - np.roll(close_1h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Pre-compute 1h volume average
    volume_1h = prices['volume'].values
    avg_volume_20 = pd.Series(volume_1h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1h > (1.3 * avg_volume_20)
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_aligned[i]) or np.isnan(atr_14[i]) or 
            np.isnan(vol_spike[i]) or np.isnan(session_filter[i])):
            signals[i] = 0.0
            continue
            
        # Skip if outside trading session
        if not session_filter[i]:
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below H3 level OR stoploss hit
            # H3 level from prior 4h bar (aligned)
            # Calculate Camarilla levels for prior completed 4h bar
            # We need the prior 4h bar's high, low, close
            # Get index of prior 4h bar in 4h data
            prior_4h_idx = i // 4  # 4 1h bars per 4h bar
            if prior_4h_idx >= 1 and prior_4h_idx < len(df_4h):
                # Get high, low, close of prior completed 4h bar
                phigh = df_4h['high'].iloc[prior_4h_idx - 1]
                plow = df_4h['low'].iloc[prior_4h_idx - 1]
                pclose = df_4h['close'].iloc[prior_4h_idx - 1]
                # Camarilla levels
                rang = phigh - plow
                h3 = pclose + (rang * 1.1 / 4)
                l3 = pclose - (rang * 1.1 / 4)
                
                if close_1h[i] < h3 or close_1h[i] < entry_price - 1.5 * atr_14[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.20
            else:
                signals[i] = 0.20  # Hold position if insufficient data for Camarilla
                
        elif position == -1:  # Short position
            # Exit: price closes above L3 level OR stoploss hit
            prior_4h_idx = i // 4
            if prior_4h_idx >= 1 and prior_4h_idx < len(df_4h):
                phigh = df_4h['high'].iloc[prior_4h_idx - 1]
                plow = df_4h['low'].iloc[prior_4h_idx - 1]
                pclose = df_4h['close'].iloc[prior_4h_idx - 1]
                rang = phigh - plow
                h3 = pclose + (rang * 1.1 / 4)
                l3 = pclose - (rang * 1.1 / 4)
                
                if close_1h[i] > l3 or close_1h[i] > entry_price + 1.5 * atr_14[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.20
            else:
                signals[i] = -0.20  # Hold position if insufficient data for Camarilla
        else:  # Flat
            # Look for Camarilla breakout with trend and volume filters
            prior_4h_idx = i // 4
            if prior_4h_idx >= 1 and prior_4h_idx < len(df_4h):
                phigh = df_4h['high'].iloc[prior_4h_idx - 1]
                plow = df_4h['low'].iloc[prior_4h_idx - 1]
                pclose = df_4h['close'].iloc[prior_4h_idx - 1]
                rang = phigh - plow
                h3 = pclose + (rang * 1.1 / 4)
                l3 = pclose - (rang * 1.1 / 4)
                
                if vol_spike[i]:
                    # Long: price breaks above H3 in uptrend (close > EMA50)
                    if close_1h[i] > h3 and close_1h[i] > ema_50_aligned[i]:
                        position = 1
                        entry_price = close_1h[i]
                        signals[i] = 0.20
                    # Short: price breaks below L3 in downtrend (close < EMA50)
                    elif close_1h[i] < l3 and close_1h[i] < ema_50_aligned[i]:
                        position = -1
                        entry_price = close_1h[i]
                        signals[i] = -0.20
    
    return signals