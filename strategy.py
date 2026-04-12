#35734
# Hypothesis: 1h strategy using 4h and 1d for direction (trend and volatility regime) and 1h for precise entry.
# Uses 4h Supertrend for trend filter, 1d ATR-based volatility regime (high/low vol) to adapt entry sensitivity,
# and 1h RSI pullback in trend direction for entry. Designed to work in bull (follow 4h trend) and bear
# (fade mean reversion in low volatility regime) by adapting to volatility state.
# Target: 15-37 trades/year (~60-150 over 4 years) to avoid fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_Supertrend_ATR_Regime_RSI_Pullback_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h Supertrend for trend filter ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate ATR(10) for Supertrend
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr_10 = np.full_like(tr, np.nan)
    if len(tr) >= 10:
        atr_sum = np.sum(tr[:10])
        atr_10[9] = atr_sum / 10
        for i in range(10, len(tr)):
            atr_sum = atr_sum - tr[i-10] + tr[i]
            atr_10[i] = atr_sum / 10
    
    # Supertrend calculation
    upper_band = np.full_like(close_4h, np.nan)
    lower_band = np.full_like(close_4h, np.nan)
    in_uptrend = np.full_like(close_4h, True)
    
    factor = 3.0
    for i in range(10, len(close_4h)):
        if np.isnan(atr_10[i]):
            continue
        upper_band[i] = (high_4h[i] + low_4h[i]) / 2 + factor * atr_10[i]
        lower_band[i] = (high_4h[i] + low_4h[i]) / 2 - factor * atr_10[i]
        
        if i == 10:
            upper_band[i] = upper_band[i]
            lower_band[i] = lower_band[i]
        else:
            if close_4h[i-1] <= upper_band[i-1]:
                upper_band[i] = min(upper_band[i], upper_band[i-1])
            else:
                upper_band[i] = upper_band[i]
            
            if close_4h[i-1] >= lower_band[i-1]:
                lower_band[i] = max(lower_band[i], lower_band[i-1])
            else:
                lower_band[i] = lower_band[i]
        
        if close_4h[i] <= upper_band[i]:
            in_uptrend[i] = True
        elif close_4h[i] >= lower_band[i]:
            in_uptrend[i] = False
        else:
            in_uptrend[i] = in_uptrend[i-1]
    
    # Align Supertrend trend to 1h
    supertrend_dir = np.where(in_uptrend, 1, -1)
    supertrend_dir_1h = align_htf_to_ltf(prices, df_4h, supertrend_dir)
    
    # === 1d ATR-based volatility regime ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR(14) for volatility
    tr1_d = high_1d - low_1d
    tr2_d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_d = np.abs(low_1d - np.roll(close_1d, 1))
    tr_d = np.maximum(tr1_d, np.maximum(tr2_d, tr3_d))
    tr_d[0] = tr1_d[0]
    atr_14 = np.full_like(tr_d, np.nan)
    if len(tr_d) >= 14:
        tr_sum = np.sum(tr_d[:14])
        atr_14[13] = tr_sum / 14
        for i in range(14, len(tr_d)):
            tr_sum = tr_sum - tr_d[i-14] + tr_d[i]
            atr_14[i] = tr_sum / 14
    
    # Calculate ATR ratio (current ATR / 50-period average) for regime
    atr_ratio = np.full_like(atr_14, np.nan)
    if len(atr_14) >= 50:
        for i in range(49, len(atr_14)):
            if np.isnan(atr_14[i]):
                continue
            start_idx = max(0, i-49)
            valid_atr = atr_14[start_idx:i+1]
            valid_atr = valid_atr[~np.isnan(valid_atr)]
            if len(valid_atr) >= 10:
                atr_mean = np.mean(valid_atr)
                if atr_mean > 0:
                    atr_ratio[i] = atr_14[i] / atr_mean
    
    # Define volatility regimes: low (<0.8), normal (0.8-1.2), high (>1.2)
    vol_regime = np.full_like(atr_ratio, 1)  # 1=normal
    vol_regime[atr_ratio < 0.8] = 0   # low volatility
    vol_regime[atr_ratio > 1.2] = 2   # high volatility
    vol_regime_1h = align_htf_to_ltf(prices, df_1d, vol_regime)
    
    # === 1h RSI for entry timing ===
    # Calculate RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(gain, np.nan)
    avg_loss = np.full_like(loss, np.nan)
    if len(gain) >= 14:
        gain_sum = np.sum(gain[:14])
        loss_sum = np.sum(loss[:14])
        avg_gain[13] = gain_sum / 14
        avg_loss[13] = loss_sum / 14
        for i in range(14, len(gain)):
            gain_sum = gain_sum - gain[i-14] + gain[i]
            loss_sum = loss_sum - loss[i-14] + loss[i]
            avg_gain[i] = gain_sum / 14
            avg_loss[i] = loss_sum / 14
    
    rs = np.full_like(avg_gain, np.nan)
    rsi = np.full_like(avg_gain, np.nan)
    for i in range(len(avg_gain)):
        if not np.isnan(avg_gain[i]) and not np.isnan(avg_loss[i]) and avg_loss[i] != 0:
            rs[i] = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs[i]))
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Generate signals
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if not in session or missing data
        if not in_session[i]:
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
            
        if (np.isnan(supertrend_dir_1h[i]) or 
            np.isnan(vol_regime_1h[i]) or 
            np.isnan(rsi[i])):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Determine entry sensitivity based on volatility regime
        # Low vol: tighter RSI levels for mean reversion
        # High vol: wider RSI levels to avoid whipsaws
        vol_regime = vol_regime_1h[i]
        if vol_regime == 0:  # Low volatility
            rsi_overbought = 65
            rsi_oversold = 35
        elif vol_regime == 2:  # High volatility
            rsi_overbought = 75
            rsi_oversold = 25
        else:  # Normal volatility
            rsi_overbought = 70
            rsi_oversold = 30
        
        # Entry conditions: RSI pullback in direction of 4h trend
        rsi_oversold_condition = rsi[i] < rsi_oversold
        rsi_overbought_condition = rsi[i] > rsi_overbought
        
        # Trend from 4h Supertrend
        trend_up = supertrend_dir_1h[i] == 1
        
        # Long: RSI oversold in uptrend OR RSI overbought in downtrend (mean reversion in low vol)
        long_entry = False
        short_entry = False
        
        if trend_up:  # Uptrend: look for long pullbacks
            long_entry = rsi_oversold_condition
        else:  # Downtrend: look for short pullbacks
            short_entry = rsi_overbought_condition
        
        # In low volatility regime, also allow mean reversion trades
        if vol_regime == 0:  # Low volatility
            if rsi_overbought_condition:
                short_entry = True  # Fade overbought
            if rsi_oversold_condition:
                long_entry = True   # Fade oversold
        
        # Exit conditions: opposite RSI signal or trend change
        long_exit = rsi[i] > 50 or supertrend_dir_1h[i] != 1
        short_exit = rsi[i] < 50 or supertrend_dir_1h[i] != -1
        
        # Signal logic
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.20
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.20
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals