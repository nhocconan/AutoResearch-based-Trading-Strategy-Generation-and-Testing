#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_keltner_channel_breakout_v2"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return signals
    
    # Calculate weekly ATR for Keltner Channels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_1w = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Calculate weekly Keltner Channels (20, 2.0)
    ema_close_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    upper_keltner = ema_close_1w + 2.0 * atr_1w
    lower_keltner = ema_close_1w - 2.0 * atr_1w
    
    # Align weekly Keltner Channels to daily timeframe
    upper_keltner_aligned = align_htf_to_ltf(prices, df_1w, upper_keltner)
    lower_keltner_aligned = align_htf_to_ltf(prices, df_1w, lower_keltner)
    
    # Volume confirmation: volume > 1.5x 20-period average (daily)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Trend filter: ADX > 25 for trending market (daily)
    plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    tr1_d = high - low
    tr2_d = np.abs(high - np.roll(close, 1))
    tr3_d = np.abs(low - np.roll(close, 1))
    tr_d = np.maximum(tr1_d, np.maximum(tr2_d, tr3_d))
    tr_d[0] = tr1_d[0]
    
    atr_d = pd.Series(tr_d).rolling(window=14, min_periods=14).mean().values
    
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / (atr_d + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / (atr_d + 1e-10)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    trending_market = adx > 25
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(upper_keltner_aligned[i]) or np.isnan(lower_keltner_aligned[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        upper_keltner = upper_keltner_aligned[i]
        lower_keltner = lower_keltner_aligned[i]
        adx_val = adx[i]
        trending = trending_market[i]
        
        # Volume confirmation
        volume_confirmed = volume_current > 1.5 * vol_ma_20[i]
        
        # Entry signals - only in trending markets
        long_signal = False
        short_signal = False
        
        # Long: price breaks above upper Keltner channel with volume and trend
        if price_high > upper_keltner and volume_confirmed and trending:
            long_signal = True
        
        # Short: price breaks below lower Keltner channel with volume and trend
        if price_low < lower_keltner and volume_confirmed and trending:
            short_signal = True
        
        # Exit conditions
        # Stop loss conditions
        stop_long = position == 1 and price_low < (entry_price - 2.0 * atr_d[i])
        stop_short = position == -1 and price_high > (entry_price + 2.0 * atr_d[i])
        
        # Exit when price returns to middle of Keltner channel (mean reversion within trend)
        middle_keltner = (upper_keltner + lower_keltner) / 2.0
        exit_long = position == 1 and price_close < middle_keltner
        exit_short = position == -1 and price_close > middle_keltner
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            entry_price = price_close
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            entry_price = price_close
            signals[i] = -0.25
        elif position == 1 and (exit_long or stop_long):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (exit_short or stop_short):
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: Keltner Channel breakout strategy with volume confirmation and ADX trend filter on daily timeframe.
# Enters long when price breaks above weekly Keltner upper channel (EMA20 + 2*ATR) with volume confirmation (>1.5x avg volume) in trending markets (ADX > 25).
# Enters short when price breaks below weekly Keltner lower channel (EMA20 - 2*ATR) with volume confirmation and ADX > 25.
# Uses weekly timeframe for Keltner Channels to capture multi-week breakouts and avoid false signals.
# Volume confirmation ensures institutional participation, ADX filter avoids whipsaws in sideways markets.
# Exits when price returns to the middle of the channel or ATR stop loss (2.0x) is hit.
# Designed for 1d timeframe with tight entry conditions to target 30-100 total trades over 4 years (7-25/year).
# Works in both bull and bear markets by trading breakouts in either direction with trend filter.