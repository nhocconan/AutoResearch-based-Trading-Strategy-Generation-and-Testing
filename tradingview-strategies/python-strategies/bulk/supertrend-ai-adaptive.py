#!/usr/bin/env python3
import numpy as np
import pandas as pd

name = "SuperTrend AI Adaptive - Strategy [BTC]"
timeframe = "4h"
leverage = 1

def numpy_sma(source, length):
    """Simple Moving Average."""
    if length <= 0:
        return np.zeros_like(source)
    pad = np.ones(length - 1) * np.nan
    extended = np.concatenate([pad, source])
    cumsum = np.cumsum(extended)
    sma = (cumsum[length:] - cumsum[:-length]) / length
    return np.concatenate([np.full(length - 1, np.nan), sma])

def numpy_rma(source, length):
    """Wilder's Smoothing (RMA)."""
    if length <= 0:
        return np.zeros_like(source)
    alpha = 1.0 / length
    rma = np.zeros_like(source)
    rma[0] = source[0]
    for i in range(1, len(source)):
        rma[i] = alpha * source[i] + (1.0 - alpha) * rma[i - 1]
    return rma

def numpy_ema(source, length):
    """Exponential Moving Average."""
    if length <= 0:
        return np.zeros_like(source)
    alpha = 2.0 / (length + 1)
    ema = np.zeros_like(source)
    ema[0] = source[0]
    for i in range(1, len(source)):
        ema[i] = alpha * source[i] + (1.0 - alpha) * ema[i - 1]
    return ema

def calculate_atr(high, low, close, length):
    """Calculate ATR using RMA of True Range."""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        prev_close = close[i - 1]
        hl = high[i] - low[i]
        hpc = abs(high[i] - prev_close)
        lpc = abs(low[i] - prev_close)
        tr[i] = max(hl, hpc, lpc)
    return numpy_rma(tr, length)

def calculate_adx(high, low, close, length):
    """Calculate ADX using RMA."""
    n = len(close)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        up_move = high[i] - high[i - 1]
        dn_move = low[i - 1] - low[i]
        
        if up_move > dn_move and up_move > 0:
            plus_dm[i] = up_move
        else:
            plus_dm[i] = 0
            
        if dn_move > up_move and dn_move > 0:
            minus_dm[i] = dn_move
        else:
            minus_dm[i] = 0
            
        prev_close = close[i - 1]
        hl = high[i] - low[i]
        hpc = abs(high[i] - prev_close)
        lpc = abs(low[i] - prev_close)
        tr[i] = max(hl, hpc, lpc)
        
    smooth_tr = numpy_rma(tr, length)
    smooth_pd = numpy_rma(plus_dm, length)
    smooth_nd = numpy_rma(minus_dm, length)
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    dx = np.zeros(n)
    
    for i in range(n):
        if smooth_tr[i] > 0:
            plus_di[i] = 100.0 * smooth_pd[i] / smooth_tr[i]
            minus_di[i] = 100.0 * smooth_nd[i] / smooth_tr[i]
        else:
            plus_di[i] = 0
            minus_di[i] = 0
            
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
        else:
            dx[i] = 0
            
    adx = numpy_rma(dx, length)
    return adx

def generate_signals(prices):
    """
    Generates trading signals based on SuperTrend AI Adaptive logic.
    Returns a numpy array of integers: 1 (Long), -1 (Short), 0 (Flat).
    """
    # Ensure inputs are numpy arrays
    open_price = prices['open'].to_numpy()
    high = prices['high'].to_numpy()
    low = prices['low'].to_numpy()
    close = prices['close'].to_numpy()
    volume = prices['volume'].to_numpy()
    n = len(close)
    
    # Parameters (Hardcoded from Pine Script Defaults)
    i_atLen = 10
    i_bMult = 3.0
    i_regLen = 40
    i_adxLen = 14
    i_adxThr = 20.0
    i_adapt = True
    i_trendLen = 50
    i_volLen = 20
    i_minSc = 65
    i_slMode = "ATR"  # Options: ATR, Percent, SuperTrend
    i_slAtr = 6.0
    i_slPct = 3.0
    i_tpMode = "RR"   # Options: RR, Percent, None
    i_tpRR = 2.5
    i_tpPct = 6.0
    i_trail = False
    i_trailAtr = 2.5
    i_trendF = True
    i_regF = True
    i_volF = True
    i_sigCD = 5
    
    # Precompute Indicators
    atr = calculate_atr(high, low, close, i_atLen)
    adx = calculate_adx(high, low, close, i_adxLen)
    atr_ma = numpy_sma(atr, i_regLen)
    trend_ema = numpy_ema(close, i_trendLen)
    vol_ma = numpy_sma(volume, i_volLen)
    hl2 = (high + low) / 2.0
    
    # State Variables
    st_band = np.zeros(n)
    st_dir = np.ones(n, dtype=np.int8)  # 1 = Bull, -1 = Bear
    signals = np.zeros(n, dtype=np.int8)
    
    # Position State
    position = 0  # 0: Flat, 1: Long, -1: Short
    entry_price = 0.0
    sl_price = 0.0
    tp_price = 0.0
    last_entry_bar = -100
    
    # Initial ST State
    prev_st_band = 0.0
    prev_st_dir = 1
    
    # Lookback buffer to avoid NaNs
    lookback = max(i_atLen, i_regLen, i_adxLen, i_trendLen, i_volLen) + 10
    
    for i in range(n):
        # Handle initial NaNs
        if i < lookback:
            st_band[i] = np.nan
            st_dir[i] = 1
            continue
            
        safe_atr = atr[i] if atr[i] > 0 else 0.001
        
        # --- Regime Detection ---
        atr_ratio = atr[i] / atr_ma[i] if atr_ma[i] > 0 else 1.0
        regime = 1
        if atr_ratio > 1.4:
            regime = 2  # Volatile
        elif adx[i] < i_adxThr and atr_ratio < 0.9:
            regime = 0  # Ranging
        else:
            regime = 1  # Trending
            
        # --- Adaptive Multiplier ---
        adapt_mult = i_bMult
        if i_adapt:
            if regime == 2:
                adapt_mult = i_bMult * (1.0 + (atr_ratio - 1.0) * 0.4)
            elif regime == 0:
                adapt_mult = i_bMult * 0.85
        adapt_mult = max(min(adapt_mult, i_bMult * 2.0), i_bMult * 0.5)
        
        # --- SuperTrend Calculation ---
        upper_base = hl2[i] + adapt_mult * atr[i]
        lower_base = hl2[i] - adapt_mult * atr[i]
        
        if prev_st_dir == 1:
            curr_band = max(lower_base, prev_st_band)
            if close[i] < curr_band:
                curr_dir = -1
                curr_band = upper_base
            else:
                curr_dir = 1
        else:
            curr_band = min(upper_base, prev_st_band)
            if close[i] > curr_band:
                curr_dir = 1
                curr_band = lower_base
            else:
                curr_dir = -1
                
        st_band[i] = curr_band
        st_dir[i] = curr_dir
        
        trend_flip = (curr_dir != prev_st_dir)
        
        # --- Filters ---
        trend_up = close[i] > trend_ema[i]
        trend_dn = close[i] < trend_ema[i]
        vol_ok = volume[i] > vol_ma[i] if vol_ma[i] > 0 else True
        cooldown_ok = (i - last_entry_bar) > i_sigCD
        
        # --- AI Scoring ---
        def calc_score(is_bull):
            score = 0.0
            # Volume
            v_rat = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 1.0
            if v_rat >= 2.5: score += 20
            elif v_rat >= 1.5: score += 14
            elif v_rat >= 1.0: score += 8
            else: score += 3
            
            # Displacement
            disp = (close[i] - curr_band) if is_bull else (curr_band - close[i])
            disp_atr = disp / safe_atr
            if disp_atr >= 1.5: score += 25
            elif disp_atr >= 0.8: score += 18
            elif disp_atr >= 0.3: score += 12
            elif disp_atr > 0: score += 5
            
            # EMA Alignment
            aligned = (is_bull and trend_up) or (not is_bull and trend_dn)
            ema_dist = abs(close[i] - trend_ema[i]) / safe_atr
            if aligned and ema_dist > 0.5: score += 20
            elif aligned: score += 14
            elif ema_dist < 0.3: score += 8
            else: score += 2
            
            # Regime
            if regime == 1: score += 15
            elif regime == 2: score += 8
            else: score += 3
            
            # Band Distance
            if i > 0 and not np.isnan(st_band[i-1]):
                prev_dist = abs(close[i-1] - st_band[i-1]) / safe_atr
                if prev_dist >= 2.0: score += 20
                elif prev_dist >= 1.0: score += 14
                elif prev_dist >= 0.5: score += 8
                else: score += 3
            
            return int(min(round(score), 100))
        
        # --- Exit Logic (Intrabar Approximation) ---
        exit_signal = False
        
        if position == 1:  # Long
            # Check SL/TP Hit using High/Low
            hit_sl = low[i] <= sl_price
            hit_tp = (i_tpMode != "None") and (high[i] >= tp_price)
            # Check ST Flip Exit
            st_flip_exit = (i_slMode == "SuperTrend") and (curr_dir == -1)
            
            if hit_sl or hit_tp or st_flip_exit:
                exit_signal = True
                
            # Trailing Stop Update
            if i_trail and not exit_signal:
                trail_dist = atr[i] * i_trailAtr
                new_sl = low[i] - trail_dist
                if new_sl > sl_price:
                    sl_price = new_sl
                    # Recheck hit after update
                    if low[i] <= sl_price:
                        exit_signal = True
                        
        elif position == -1:  # Short
            hit_sl = high[i] >= sl_price
            hit_tp = (i_tpMode != "None") and (low[i] <= tp_price)
            st_flip_exit = (i_slMode == "SuperTrend") and (curr_dir == 1)
            
            if hit_sl or hit_tp or st_flip_exit:
                exit_signal = True
                
            if i_trail and not exit_signal:
                trail_dist = atr[i] * i_trailAtr
                new_sl = high[i] + trail_dist
                if new_sl < sl_price:
                    sl_price = new_sl
                    if high[i] >= sl_price:
                        exit_signal = True
                        
        if exit_signal:
            position = 0
            entry_price = 0.0
            sl_price = 0.0
            tp_price = 0.0
            
        # --- Entry Logic ---
        enter_long = False
        enter_short = False
        
        if trend_flip and cooldown_ok:
            if curr_dir == 1:  # Bullish Flip
                sig_score = calc_score(True)
                pass_score = sig_score >= i_minSc
                pass_trend = (not i_trendF) or trend_up
                pass_reg = (not i_regF) or (regime != 0)
                pass_vol = (not i_volF) or vol_ok
                
                if pass_score and pass_trend and pass_reg and pass_vol:
                    enter_long = True
                    
            else:  # Bearish Flip
                sig_score = calc_score(False)
                pass_score = sig_score >= i_minSc
                pass_trend = (not i_trendF) or trend_dn
                pass_reg = (not i_regF) or (regime != 0)
                pass_vol = (not i_volF) or vol_ok
                
                if pass_score and pass_trend and pass_reg and pass_vol:
                    enter_short = True
                    
        # Execute Entry (if not exited same bar)
        if position == 0:
            if enter_long:
                position = 1
                entry_price = close[i]
                # Calc SL
                if i_slMode == "ATR":
                    sl_dist = atr[i] * i_slAtr
                elif i_slMode == "Percent":
                    sl_dist = close[i] * i_slPct / 100.0
                else: # SuperTrend
                    sl_dist = abs(close[i] - curr_band)
                    
                sl_price = close[i] - sl_dist
                
                # Calc TP
                if i_tpMode == "RR":
                    tp_dist = sl_dist * i_tpRR
                elif i_tpMode == "Percent":
                    tp_dist = close[i] * i_tpPct / 100.0
                else:
                    tp_dist = 0.0
                    
                tp_price = close[i] + tp_dist if i_tpMode != "None" else np.nan
                last_entry_bar = i
                
            elif enter_short:
                position = -1
                entry_price = close[i]
                if i_slMode == "ATR":
                    sl_dist = atr[i] * i_slAtr
                elif i_slMode == "Percent":
                    sl_dist = close[i] * i_slPct / 100.0
                else:
                    sl_dist = abs(close[i] - curr_band)
                    
                sl_price = close[i] + sl_dist
                
                if i_tpMode == "RR":
                    tp_dist = sl_dist * i_tpRR
                elif i_tpMode == "Percent":
                    tp_dist = close[i] * i_tpPct / 100.0
                else:
                    tp_dist = 0.0
                    
                tp_price = close[i] - tp_dist if i_tpMode != "None" else np.nan
                last_entry_bar = i
        
        # Set Signal
        signals[i] = position
        
        # Update State for next iteration
        prev_st_band = curr_band
        prev_st_dir = curr_dir
        
    return signals

if __name__ == "__main__":
    # Example usage stub
    pass
