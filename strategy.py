#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h ADX(14) trend filter with Donchian(20) breakout and volume confirmation.
# In strong trends (ADX > 25): follow breakouts (long on upper band break, short on lower band break).
# In weak trends/ranges (ADX <= 25): mean-revert at Donchian bands (short near upper band, long near lower band).
# Uses volume > 1.3x 20-period average for confirmation. Avoids whipsaws in weak trends.
# Target: 20-50 trades/year by requiring ADX alignment + breakout/reversion + volume confirmation.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Calculate True Range and ATR for ADX
    tr_list = []
    for i in range(n):
        if i == 0:
            tr_list.append(0)
        else:
            high_low = prices['high'].iloc[i] - prices['low'].iloc[i]
            high_close = abs(prices['high'].iloc[i] - prices['close'].iloc[i-1])
            low_close = abs(prices['low'].iloc[i] - prices['close'].iloc[i-1])
            tr_list.append(max(high_low, high_close, low_close))
    
    tr_series = pd.Series(tr_list)
    atr = tr_series.rolling(window=14, min_periods=14).mean()
    
    # Calculate +DM and -DM
    up_move = prices['high'].diff()
    down_move = prices['low'].diff().multiply(-1)
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Calculate smoothed +DM and -DM
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean()
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean()
    
    # Calculate +DI and -DI
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr
    
    # Calculate DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=14, min_periods=14).mean()
    adx_values = adx.fillna(0).values  # fill NaN with 0 for weak trend
    
    # Pre-compute volume moving average (20-period)
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(28, n):  # start after ADX warmup
        # Skip if data not ready
        if np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Donchian channels (20-period)
        lookback_start = max(0, i - 19)
        high_window = prices['high'].iloc[lookback_start:i+1].values
        low_window = prices['low'].iloc[lookback_start:i+1].values
        donchian_high = np.max(high_window)
        donchian_low = np.min(low_window)
        
        # Current price and volume
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirm = volume > 1.3 * vol_ma[i]
        
        # Trend classification using ADX
        is_strong_trend = adx_values[i] > 25
        is_weak_trend = adx_values[i] <= 25
        
        if position == 0:
            if is_strong_trend and volume_confirm:
                # Strong trend: follow breakouts
                if price > donchian_high:
                    signals[i] = 0.25
                    position = 1
                elif price < donchian_low:
                    signals[i] = -0.25
                    position = -1
            elif is_weak_trend and volume_confirm:
                # Weak trend/range: mean revert at extremes
                # Short near upper band, long near lower band
                if price >= donchian_high * 0.995:  # within 0.5% of upper band
                    signals[i] = -0.25
                    position = -1
                elif price <= donchian_low * 1.005:  # within 0.5% of lower band
                    signals[i] = 0.25
                    position = 1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                if is_strong_trend:
                    # Exit long on breakdown below Donchian low in strong trend
                    if price < donchian_low:
                        exit_signal = True
                else:  # weak trend
                    # Exit long when price moves to middle of range or hits upper band
                    if price >= donchian_high * 0.995:  # near upper band
                        exit_signal = True
            
            elif position == -1:  # short position
                if is_strong_trend:
                    # Exit short on breakout above Donchian high in strong trend
                    if price > donchian_high:
                        exit_signal = True
                else:  # weak trend
                    # Exit short when price moves to middle of range or hits lower band
                    if price <= donchian_low * 1.005:  # near lower band
                        exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_ADX_Donchian_BreakoutMeanRev_Volume"
timeframe = "4h"
leverage = 1.0