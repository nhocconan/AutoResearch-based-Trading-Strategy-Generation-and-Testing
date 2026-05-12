# 4H_CAMARILLA_R1S1_BREAKOUT_VOLUME_CONFIRMATION
# Strategy: Camarilla pivot breakout on R1/S1 levels with volume confirmation and 1d trend filter
# Works in bull/bear: breakouts capture momentum, volume filters fakeouts, trend filter avoids counter-trend trades
# Target: 25-40 trades/year per symbol (~100-160 total over 4 years)
# Edge: Camarilla levels derived from actual institutional algorithms, volume confirms institutional participation
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4H_CAMARILLA_R1S1_BREAKOUT_VOLUME_CONFIRMATION"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1D TREND FILTER (HTF) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # === CAMARILLA PIVOTS FROM PREVIOUS DAY ===
    # Calculate from previous day's OHLC (available at 4h open)
    prev_high = np.roll(high, 1)  # previous bar's high
    prev_low = np.roll(low, 1)    # previous bar's low
    prev_close = np.roll(close, 1) # previous bar's close
    
    # For 4h data, we need previous day's values, not previous bar
    # But since we don't have easy day grouping, we approximate with 6-period lookback (6*4h=24h)
    # Actually, better: use daily data directly for pivot calculation
    # We'll calculate pivots from daily OHLC and align to 4h
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    # Actually, standard Camarilla uses: (H+L+C)/3 as pivot
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    R1_1d = pivot_1d + (range_1d * 1.0 / 12)
    S1_1d = pivot_1d - (range_1d * 1.0 / 12)
    
    # Align to 4h
    R1_1d_aligned = align_htf_to_ltf(prices, df_1d, R1_1d)
    S1_1d_aligned = align_htf_to_ltf(prices, df_1d, S1_1d)
    
    # === VOLUME FILTER ===
    # Volume spike: current volume > 1.5 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # need both 1d EMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if 1d trend data not ready
        if np.isnan(ema34_1d_aligned[i]) or np.isnan(R1_1d_aligned[i]) or np.isnan(S1_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 with volume spike + 1d uptrend
            if (close[i] > R1_1d_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume spike + 1d downtrend
            elif (close[i] < S1_1d_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below pivot OR 1d trend flips
            if (close[i] < pivot_1d_aligned[i] or 
                close[i] < ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above pivot OR 1d trend flips
            if (close[i] > pivot_1d_aligned[i] or 
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals