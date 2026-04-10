#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R extreme with 1d volume spike and 1w ADX regime filter
# - Long when Williams %R crosses above -80 from below (oversold bounce) with volume > 1.5x 20-day EMA and 1w ADX > 20
# - Short when Williams %R crosses below -20 from above (overbought rejection) with volume > 1.5x 20-day EMA and 1w ADX > 20
# - Exit: ATR trailing stop (2.0x ATR) or Williams %R returns to neutral zone (-50)
# - Position sizing: 0.25 discrete level
# - Targets ~20-30 trades/year on 4h timeframe. Williams %R captures mean reversion in extended moves,
#   volume confirmation validates the reversal, 1w ADX > 20 ensures sufficient trend strength for follow-through.
#   Works in bull/bear: oversold bounces in bear markets, overbought rejections in bull markets, regime filter avoids chop.

name = "4h_1d_1w_williamsr_volume_adx_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute HTF data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Calculate Williams %R(14) on 4h
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)  # handle division by zero
    
    # Calculate 1d volume EMA for confirmation (20-period)
    volume_ema_20_1d = pd.Series(volume_1d).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ema_20_1d)
    
    # Calculate 1w ADX(14) for trend filter
    high_1w_series = pd.Series(high_1w)
    low_1w_series = pd.Series(low_1w)
    close_1w_series = pd.Series(close_1w)
    
    up_move = high_1w_series.diff()
    down_move = low_1w_series.diff().mul(-1)
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr2[0] = tr2[1] if len(tr2) > 1 else 0
    tr3[0] = tr3[1] if len(tr3) > 1 else 0
    tr_1w = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_1w = pd.Series(tr_1w).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, min_periods=14, adjust=False).mean().values / atr_1w
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, min_periods=14, adjust=False).mean().values / atr_1w
    
    dx = np.abs(plus_di - minus_di) / (np.abs(plus_di) + np.abs(minus_di)) * 100
    dx = np.where(np.isnan(dx), 0, dx)
    adx_1w = pd.Series(dx).ewm(span=14, min_periods=14, adjust=False).mean().values
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Calculate ATR(14) for trailing stop on 4h
    tr1_4h = high - low
    tr2_4h = np.abs(high - np.roll(close, 1))
    tr3_4h = np.abs(low - np.roll(close, 1))
    tr2_4h[0] = tr2_4h[1] if len(tr2_4h) > 1 else 0
    tr3_4h[0] = tr3_4h[1] if len(tr3_4h) > 1 else 0
    tr_4h = np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))
    atr_4h = pd.Series(tr_4h).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(volume_ema_20_1d_aligned[i]) or 
            np.isnan(adx_1w_aligned[i]) or np.isnan(atr_4h[i])):
            signals[i] = 0.0
            continue
        
        # Get current 1d volume for confirmation (aligned to 4h)
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)
        vol_confirm = vol_1d_current[i] > 1.5 * volume_ema_20_1d_aligned[i]
        
        # Regime filter: 1w ADX > 20 indicates sufficient trend strength
        regime_filter = adx_1w_aligned[i] > 20
        
        # Williams %R entry conditions (crossing extreme levels)
        # Long: Williams %R crosses above -80 from below (oversold bounce)
        # Short: Williams %R crosses below -20 from above (overbought rejection)
        williams_r_prev = williams_r[i-1] if i > 0 else -50
        
        long_entry = (williams_r[i] > -80 and williams_r_prev <= -80 and  # Cross above -80
                     vol_confirm and 
                     regime_filter)
        short_entry = (williams_r[i] < -20 and williams_r_prev >= -20 and  # Cross below -20
                      vol_confirm and 
                      regime_filter)
        
        if position == 0:  # Flat - look for entry
            if long_entry:
                position = 1
                signals[i] = 0.25
                highest_since_entry = close[i]
                lowest_since_entry = close[i]
            elif short_entry:
                position = -1
                signals[i] = -0.25
                highest_since_entry = close[i]
                lowest_since_entry = close[i]
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Update highest/lowest since entry
            if position == 1:
                highest_since_entry = max(highest_since_entry, close[i])
                lowest_since_entry = min(lowest_since_entry, close[i])
                # ATR trailing stop: exit if price drops 2.0*ATR from high
                # or Williams %R returns to neutral zone (> -50)
                if (close[i] < highest_since_entry - 2.0 * atr_4h[i] or  # trailing stop
                    williams_r[i] > -50):         # Williams %R mean reversion exit
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                highest_since_entry = max(highest_since_entry, close[i])
                lowest_since_entry = min(lowest_since_entry, close[i])
                # ATR trailing stop: exit if price rises 2.0*ATR from low
                # or Williams %R returns to neutral zone (< -50)
                if (close[i] > lowest_since_entry + 2.0 * atr_4h[i] or  # trailing stop
                    williams_r[i] < -50):         # Williams %R mean reversion exit
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals