#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 12h volume confirmation and 1d trend filter
# - Long when price breaks above Camarilla H3 (4h) AND 12h volume > 2.0x 20-bar avg AND 1d EMA(50) > EMA(200)
# - Short when price breaks below Camarilla L3 (4h) AND 12h volume > 2.0x 20-bar avg AND 1d EMA(50) < EMA(200)
# - Exit when price returns to Camarilla pivot point (4h) for mean reversion
# - Uses discrete position sizing (0.25) to limit fee churn and manage drawdown
# - Camarilla levels from 4h timeframe provide precise intraday support/resistance
# - 12h volume confirmation ensures institutional participation
# - 1d EMA filter aligns with higher timeframe trend to avoid counter-trend trades
# - Target: 75-200 total trades over 4 years (19-50/year) on 4h timeframe
# - Works in both bull and bear markets: breakouts in trends, mean reversion in ranges

name = "4h_12h_1d_camarilla_breakout_volume_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 12h volume confirmation: > 2.0x 20-period average
    volume_12h = df_12h['volume'].values
    volume_20_avg_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_spike_12h = volume_12h > (2.0 * volume_20_avg_12h)
    
    # Pre-compute 1d EMA trend filter: EMA(50) vs EMA(200)
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_bullish_1d = ema_50_1d > ema_200_1d
    ema_bearish_1d = ema_50_1d < ema_200_1d
    
    # Pre-compute 4h Camarilla pivot levels
    high_4h = df_12h['high'].values  # Using 12h data for Camarilla? No, need 4h data
    # Correction: Need actual 4h data for Camarilla calculations
    # Since we're on 4h timeframe, we need to get 4h OHLC for Camarilla
    # But we can't call get_htf_data for same timeframe - we'll use prices directly for 4h Camarilla
    # However, we need to calculate Camarilla from completed 4h bars
    # Let's get 4h data by using the prices dataframe but ensuring we use completed bars
    # Actually, for Camarilla we need the prior completed bar's OHLC
    # We'll calculate Camarilla on 4h timeframe using the prices dataframe but shift by 1 to avoid lookahead
    
    # For 4h timeframe strategy, we need to calculate Camarilla from 4h data
    # Since we're already on 4h timeframe, we can use prices directly but need to ensure we use completed bars
    # We'll calculate Camarilla using the prior completed 4h bar's data
    
    # Let's get the 4h OHLC from prices (our primary timeframe)
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    # But we must use only completed bars - so we shift by 1 to avoid using current bar's data
    # However, Camarilla levels are calculated from the prior bar's range
    # So we calculate Camarilla for bar i using bar i-1's OHLC
    high_4h_prev = np.roll(high_4h, 1)
    low_4h_prev = np.roll(low_4h, 1)
    close_4h_prev = np.roll(close_4h, 1)
    # Set first value to NaN since there's no prior bar
    high_4h_prev[0] = np.nan
    low_4h_prev[0] = np.nan
    close_4h_prev[0] = np.nan
    
    # Camarilla calculations using prior bar's data
    rang = high_4h_prev - low_4h_prev
    camarilla_pivot = (high_4h_prev + low_4h_prev + close_4h_prev) / 3
    camarilla_h3 = camarilla_pivot + (rang * 1.1 / 4)
    camarilla_l3 = camarilla_pivot - (rang * 1.1 / 4)
    camarilla_h4 = camarilla_pivot + (rang * 1.1 / 2)
    camarilla_l4 = camarilla_pivot - (rang * 1.1 / 2)
    
    # Align HTF indicators to 4h timeframe
    vol_spike_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_spike_12h)
    ema_bullish_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_bullish_1d)
    ema_bearish_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_bearish_1d)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_12h, camarilla_pivot)  # Camarilla is 4h-based but we align using 12h as reference? No
    # Correction: Camarilla levels are calculated on 4h timeframe, so we need to align them properly
    # Since we're on 4h timeframe and Camarilla is 4h-based, we don't need HTF alignment for Camarilla
    # But we calculated it using rolled arrays which may have alignment issues
    # Let's simplify: calculate Camarilla directly and use as-is since we're on 4h timeframe
    # Actually, we need to ensure we're using completed Camarilla levels
    # The roll(1) ensures we use prior bar's data, which is completed
    
    # For safety, let's treat Camarilla as LTf indicators since we're calculating on 4h data
    camarilla_pivot_aligned = camarilla_pivot
    camarilla_h3_aligned = camarilla_h3
    camarilla_l3_aligned = camarilla_l3
    camarilla_h4_aligned = camarilla_h4
    camarilla_l4_aligned = camarilla_l4
    
    # Session filter: 08-20 UTC (avoid low liquidity Asian session)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(vol_spike_12h_aligned[i]) or np.isnan(ema_bullish_1d_aligned[i]) or
            np.isnan(ema_bearish_1d_aligned[i]) or np.isnan(camarilla_pivot_aligned[i]) or
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Apply session filter
        if not in_session[i]:
            # Outside session: flatten position
            position = 0
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above H3 AND 12h volume spike AND 1d bullish trend
            if (prices['close'].iloc[i] > camarilla_h3_aligned[i] and 
                vol_spike_12h_aligned[i] and 
                ema_bullish_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short when price breaks below L3 AND 12h volume spike AND 1d bearish trend
            elif (prices['close'].iloc[i] < camarilla_l3_aligned[i] and 
                  vol_spike_12h_aligned[i] and 
                  ema_bearish_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit to pivot point (mean reversion)
            # Exit when price returns to Camarilla pivot point
            exit_long = position == 1 and prices['close'].iloc[i] <= camarilla_pivot_aligned[i]
            exit_short = position == -1 and prices['close'].iloc[i] >= camarilla_pivot_aligned[i]
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals