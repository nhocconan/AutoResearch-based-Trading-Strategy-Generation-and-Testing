# 4h_12h_Camarilla_Breakout_Momentum_v1
# Hypothesis: 4h price breakouts at 12h Camarilla H3/L3 levels with 12h momentum confirmation (RSI > 55 for long, < 45 for short) and volume filter (>1.3x 20-period average). 
# Uses 12h timeframe for structure (Camarilla levels, momentum) and 4h for entry timing. Designed for 20-40 trades/year per symbol with clear trend momentum bias that works in bull (breakouts continue) and bear (failed breaks reverse) markets.
# Momentum filter avoids choppy breakouts, volume confirmation avoids low-liquidity false signals.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_Camarilla_Breakout_Momentum_v1"
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
    
    # === 12H DATA FOR CAMARILLA AND MOMENTUM ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h RSI (14-period) for momentum
    delta = np.diff(close_12h, prepend=close_12h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing for RSI
    def wilders_rsi(gain, loss, period=14):
        avg_gain = np.full_like(gain, np.nan)
        avg_loss = np.full_like(loss, np.nan)
        if len(gain) < period:
            return np.full_like(gain, 50.0)  # neutral when insufficient data
        avg_gain[period-1] = np.mean(gain[:period])
        avg_loss[period-1] = np.mean(loss[:period])
        for i in range(period, len(gain)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_12h = wilders_rsi(gain, loss, 14)
    rsi_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_12h)
    
    # === 12H CAMARILLA LEVELS FROM PREVIOUS DAY ===
    # Map each 12h bar to previous day's OHLC for pivot calculation
    pivots_high = np.full(n, np.nan)
    pivots_low = np.full(n, np.nan)
    pivots_close = np.full(n, np.nan)
    
    for i in range(n):
        current_date = pd.Timestamp(prices.iloc[i]['open_time']).date()
        prev_date = current_date - pd.Timedelta(days=1)
        
        # Find previous day in 12h data (need daily data, so use 1d)
        # Get 1d data for proper daily OHLC
        df_1d = get_htf_data(prices, '1d')
        if len(df_1d) == 0:
            continue
            
        prev_day_idx = None
        for j in range(len(df_1d)):
            if pd.Timestamp(df_1d.iloc[j]['open_time']).date() == prev_date:
                prev_day_idx = j
                break
        
        if prev_day_idx is not None and len(df_1d) > prev_day_idx:
            ph = df_1d['high'].iloc[prev_day_idx]
            pl = df_1d['low'].iloc[prev_day_idx]
            pc = df_1d['close'].iloc[prev_day_idx]
            
            pivots_high[i] = ph
            pivots_low[i] = pl
            pivots_close[i] = pc
    
    # Calculate Camarilla H3 and L3 levels
    H3 = pivots_close + (pivots_high - pivots_low) * 1.1 / 4
    L3 = pivots_close - (pivots_high - pivots_low) * 1.1 / 4
    
    # Volume filter (4h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if not ready
        if (np.isnan(H3[i]) or np.isnan(L3[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(rsi_12h_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Momentum filters: RSI > 55 for long bias, < 45 for short bias
        long_momentum = rsi_12h_aligned[i] > 55
        short_momentum = rsi_12h_aligned[i] < 45
        
        # Volume strength
        strong_volume = volume[i] > (vol_ma[i] * 1.3)
        
        # Long: price breaks above H3 with volume and bullish momentum
        long_signal = (close[i] > H3[i] and 
                      strong_volume and 
                      long_momentum)
        
        # Short: price breaks below L3 with volume and bearish momentum
        short_signal = (close[i] < L3[i] and 
                       strong_volume and 
                       short_momentum)
        
        # Exit: price returns to pivot level or momentum reverses
        exit_long = (position == 1 and 
                    (close[i] < pivots_close[i] or rsi_12h_aligned[i] < 50))
        exit_short = (position == -1 and 
                     (close[i] > pivots_close[i] or rsi_12h_aligned[i] > 50))
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals