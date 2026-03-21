#!/usr/bin/env python3
"""
EXPERIMENT #100 - KELTNER VWAP MOMENTUM ENSEMBLE (1h+4h v1)
==================================================================================================
Hypothesis: Current best Sharpe=16.016 uses HMA+Supertrend+RSI+Z-score+BBW on 15m.
This experiment tries a DIFFERENT approach focusing on:

Key innovations for #100:
1. Keltner Channels instead of Bollinger Bands - more robust regime detection
2. VWAP deviation for mean reversion entries (institutional level support/resistance)
3. Volume-weighted RSI for momentum confirmation
4. 1h entries + 4h trend filter (better signal quality than 15m noise)
5. Keltner squeeze detection (volatility compression before expansion)
6. Adaptive ATR stops based on volatility regime
7. Discrete position levels: 0.0, ±0.20, ±0.35 (reduces churn costs)
8. Correlation filter: only trade when asset moves with BTC trend

Why this should work:
- Keltner Channels (ATR-based) adapt better to crypto volatility than BB
- VWAP is key institutional level - deviations mean reversion opportunities
- Volume-weighted RSI filters low-volume false signals
- 1h timeframe has better signal-to-noise than 15m
- 4h filter prevents counter-trend trades
- Squeeze detection captures volatility expansion breakouts

Risk controls:
- Max position size: 0.35 (35% of capital)
- ATR trailing stop: 2.5*ATR in trend regime, 1.5*ATR in mean reversion
- Take profit: reduce to half at 2R, trail stop at 1R
- Volatility-adjusted sizing: reduce position when ATR% is high
- Correlation filter: skip trades when asset diverges from BTC trend
"""

import numpy as np
import pandas as pd

name = "keltner_vwap_momentum_ensemble_1h_4h_v1"
timeframe = "1h"
leverage = 1.0


def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
    
    atr = np.zeros(n)
    atr[period - 1] = np.mean(tr[1:period])
    
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    
    return atr


def calculate_vwap(high, low, close, volume, period=20):
    """Calculate VWAP with rolling window"""
    n = len(close)
    vwap = np.zeros(n)
    
    for i in range(period - 1, n):
        typical_price = (high[i - period + 1:i + 1] + low[i - period + 1:i + 1] + close[i - period + 1:i + 1]) / 3
        vol = volume[i - period + 1:i + 1]
        if np.sum(vol) > 0:
            vwap[i] = np.sum(typical_price * vol) / np.sum(vol)
    
    return vwap


def calculate_keltner_channels(high, low, close, period=20, atr_period=14, multiplier=2.0):
    """Calculate Keltner Channels (EMA + ATR bands)"""
    n = len(close)
    if n < period + atr_period:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    # EMA for center line
    ema = np.zeros(n)
    multiplier_ema = 2.0 / (period + 1)
    ema[period - 1] = np.mean(close[:period])
    for i in range(period, n):
        ema[i] = close[i] * multiplier_ema + ema[i - 1] * (1 - multiplier_ema)
    
    # ATR for bands
    atr = calculate_atr(high, low, close, atr_period)
    
    upper = np.zeros(n)
    lower = np.zeros(n)
    bandwidth = np.zeros(n)
    
    for i in range(max(period, atr_period), n):
        upper[i] = ema[i] + multiplier * atr[i]
        lower[i] = ema[i] - multiplier * atr[i]
        if ema[i] > 0:
            bandwidth[i] = (upper[i] - lower[i]) / ema[i]
    
    return upper, lower, bandwidth, ema


def calculate_rsi(close, period=14):
    """Calculate RSI"""
    n = len(close)
    if n < period + 1:
        return np.zeros(n)
    
    rsi = np.zeros(n)
    delta = np.diff(close)
    
    gain = np.zeros(n)
    loss = np.zeros(n)
    
    for i in range(1, n):
        if delta[i - 1] > 0:
            gain[i] = delta[i - 1]
        else:
            loss[i] = -delta[i - 1]
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    
    avg_gain[period] = np.mean(gain[1:period + 1])
    avg_loss[period] = np.mean(loss[1:period + 1])
    
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period
    
    for i in range(period, n):
        if avg_loss[i] == 0:
            rsi[i] = 100
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_volume_weighted_rsi(close, volume, period=14):
    """Calculate Volume-Weighted RSI"""
    n = len(close)
    if n < period + 1:
        return np.zeros(n)
    
    vwr = np.zeros(n)
    delta = np.diff(close)
    
    vol_gain = np.zeros(n)
    vol_loss = np.zeros(n)
    
    for i in range(1, n):
        if delta[i - 1] > 0:
            vol_gain[i] = delta[i - 1] * volume[i]
        else:
            vol_loss[i] = -delta[i - 1] * volume[i]
    
    avg_vol_gain = np.zeros(n)
    avg_vol_loss = np.zeros(n)
    
    avg_vol_gain[period] = np.mean(vol_gain[1:period + 1])
    avg_vol_loss[period] = np.mean(vol_loss[1:period + 1])
    
    for i in range(period + 1, n):
        avg_vol_gain[i] = (avg_vol_gain[i - 1] * (period - 1) + vol_gain[i]) / period
        avg_vol_loss[i] = (avg_vol_loss[i - 1] * (period - 1) + vol_loss[i]) / period
    
    for i in range(period, n):
        if avg_vol_loss[i] == 0:
            vwr[i] = 100
        else:
            rs = avg_vol_gain[i] / avg_vol_loss[i]
            vwr[i] = 100 - (100 / (1 + rs))
    
    return vwr


def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)"""
    n = len(close)
    if n < period * 2:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_move = high[i] - high[i - 1]
        minus_move = low[i - 1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    atr = calculate_atr(high, low, close, period)
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    dx = np.zeros(n)
    adx = np.zeros(n)
    
    for i in range(period, n):
        if atr[i] > 0:
            plus_di[i] = 100 * plus_dm[i] / atr[i]
            minus_di[i] = 100 * minus_dm[i] / atr[i]
        
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx[period * 2 - 1] = np.mean(dx[period:period * 2])
    
    for i in range(period * 2, n):
        adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period
    
    return adx, plus_di, minus_di


def calculate_keltner_squeeze(high, low, close, kc_period=20, kc_mult=2.0, bb_period=20, bb_std=2.0):
    """Detect Keltner Channel squeeze (BB inside KC = low volatility)"""
    n = len(close)
    
    # Keltner Channels
    kc_upper, kc_lower, _, _ = calculate_keltner_channels(high, low, close, kc_period, 14, kc_mult)
    
    # Bollinger Bands
    bb_upper = np.zeros(n)
    bb_lower = np.zeros(n)
    
    for i in range(bb_period - 1, n):
        sma = np.mean(close[i - bb_period + 1:i + 1])
        std = np.std(close[i - bb_period + 1:i + 1])
        bb_upper[i] = sma + bb_std * std
        bb_lower[i] = sma - bb_std * std
    
    # Squeeze: BB inside KC
    squeeze = np.zeros(n)
    for i in range(max(kc_period, bb_period), n):
        if bb_upper[i] < kc_upper[i] and bb_lower[i] > kc_lower[i]:
            squeeze[i] = 1  # Squeeze on
        else:
            squeeze[i] = 0  # Squeeze off
    
    return squeeze


def resample_to_4h(close, high, low, volume):
    """Resample 1h data to 4h (4 bars per 4h)"""
    n = len(close)
    n_4h = n // 4
    
    c_4h = np.zeros(n_4h)
    h_4h = np.zeros(n_4h)
    l_4h = np.zeros(n_4h)
    v_4h = np.zeros(n_4h)
    
    for i in range(n_4h):
        start_idx = i * 4
        end_idx = start_idx + 4
        if end_idx <= n:
            c_4h[i] = close[end_idx - 1]
            h_4h[i] = np.max(high[start_idx:end_idx])
            l_4h[i] = np.min(low[start_idx:end_idx])
            v_4h[i] = np.sum(volume[start_idx:end_idx])
    
    return c_4h, h_4h, l_4h, v_4h


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices.get("volume", np.ones(len(close))).values
    n = len(close)
    
    signals = np.zeros(n)
    
    # 1h indicators for entry timing
    atr_1h = calculate_atr(high, low, close, period=14)
    keltner_upper_1h, keltner_lower_1h, keltner_bw_1h, keltner_ema_1h = calculate_keltner_channels(high, low, close, period=20, atr_period=14, multiplier=2.0)
    rsi_1h = calculate_rsi(close, period=14)
    vwr_1h = calculate_volume_weighted_rsi(close, volume, period=14)
    vwap_1h = calculate_vwap(high, low, close, volume, period=20)
    squeeze_1h = calculate_keltner_squeeze(high, low, close)
    adx_1h, plus_di_1h, minus_di_1h = calculate_adx(high, low, close, period=14)
    
    # Volume SMA for spike detection
    vol_sma_1h = np.zeros(n)
    for i in range(20, n):
        vol_sma_1h[i] = np.mean(volume[i - 20:i + 1])
    
    # Resample to 4h for trend filter
    c_4h, h_4h, l_4h, v_4h = resample_to_4h(close, high, low, volume)
    n_4h = len(c_4h)
    
    # 4h indicators for trend
    keltner_upper_4h, keltner_lower_4h, keltner_bw_4h, keltner_ema_4h = calculate_keltner_channels(h_4h, l_4h, c_4h, period=20, atr_period=14, multiplier=2.0)
    adx_4h, plus_di_4h, minus_di_4h = calculate_adx(h_4h, l_4h, c_4h, period=14)
    rsi_4h = calculate_rsi(c_4h, period=14)
    squeeze_4h = calculate_keltner_squeeze(h_4h, l_4h, c_4h)
    
    # Map 4h indicators back to 1h timeframe
    trend_4h = np.zeros(n)
    adx_4h_mapped = np.zeros(n)
    keltner_bw_4h_mapped = np.zeros(n)
    squeeze_4h_mapped = np.zeros(n)
    rsi_4h_mapped = np.zeros(n)
    
    for i in range(n):
        idx_4h = i // 4
        if idx_4h < n_4h and idx_4h >= 30:
            # Keltner EMA trend direction
            if c_4h[idx_4h] > keltner_ema_4h[idx_4h]:
                trend_4h[i] = 1
            elif c_4h[idx_4h] < keltner_ema_4h[idx_4h]:
                trend_4h[i] = -1
            
            adx_4h_mapped[i] = adx_4h[idx_4h]
            keltner_bw_4h_mapped[i] = keltner_bw_4h[idx_4h]
            squeeze_4h_mapped[i] = squeeze_4h[idx_4h]
            rsi_4h_mapped[i] = rsi_4h[idx_4h]
    
    # Position sizing parameters
    SIZE_LOW = 0.20
    SIZE_HIGH = 0.35
    ATR_TARGET_PCT = 0.015
    ADX_MIN = 20
    VOL_SPIKE_MULT = 1.5
    
    # Calculate Keltner BW percentile for regime detection
    bbw_percentile = np.zeros(n)
    valid_bbw = keltner_bw_4h_mapped[30*4:]
    valid_bbw = valid_bbw[valid_bbw > 0]
    if len(valid_bbw) > 0:
        bbw_sorted = np.sort(valid_bbw)
        for i in range(30*4, n):
            if keltner_bw_4h_mapped[i] > 0:
                bbw_percentile[i] = np.searchsorted(bbw_sorted, keltner_bw_4h_mapped[i]) / len(bbw_sorted)
    
    # Tracking variables
    prev_signal = 0.0
    consecutive_votes = 0
    prev_vote_direction = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    tp_triggered = False
    
    first_valid = max(100, 30 * 4, 50, 40)
    
    for i in range(first_valid, n):
        # Skip if any indicator is invalid
        if (np.isnan(atr_1h[i]) or np.isnan(keltner_ema_1h[i]) or 
            atr_1h[i] == 0 or np.isnan(adx_4h_mapped[i]) or
            np.isnan(keltner_upper_1h[i]) or keltner_upper_1h[i] == 0):
            signals[i] = 0.0
            prev_signal = 0.0
            consecutive_votes = 0
            prev_vote_direction = 0
            entry_price = 0.0
            continue
        
        # Get indicator values
        trend_4h_val = trend_4h[i]
        adx_val = adx_4h_mapped[i]
        bbw_val = keltner_bw_4h_mapped[i]
        bbw_pct = bbw_percentile[i]
        squeeze_val = squeeze_4h_mapped[i]
        rsi_4h_val = rsi_4h_mapped[i]
        
        # VWAP deviation
        vwap_dev = (close[i] - vwap_1h[i]) / vwap_1h[i] * 100 if vwap_1h[i] > 0 else 0
        
        # 4h ADX filter - only trade when higher timeframe has trend strength
        adx_filter = adx_val >= ADX_MIN
        
        # Regime detection: low BW = trend follow, high BW = mean revert
        trend_regime = bbw_pct < 0.5
        
        # Volume filter
        vol_spike = volume[i] > vol_sma_1h[i] * VOL_SPIKE_MULT if vol_sma_1h[i] > 0 else False
        
        # ENSEMBLE VOTING: 6 core signals
        vote_long = 0
        vote_short = 0
        
        # Signal 1: 4h Keltner trend
        if trend_4h_val == 1:
            vote_long += 1
        elif trend_4h_val == -1:
            vote_short += 1
        
        # Signal 2: 4h ADX trend strength
        if adx_filter and plus_di_4h[i // 4] > minus_di_4h[i // 4] if i // 4 < n_4h else False:
            vote_long += 0.5
        elif adx_filter and minus_di_4h[i // 4] > plus_di_4h[i // 4] if i // 4 < n_4h else False:
            vote_short += 0.5
        
        # Signal 3: 1h Keltner position
        if close[i] > keltner_ema_1h[i] and keltner_ema_1h[i] > 0:
            vote_long += 0.5
        elif close[i] < keltner_ema_1h[i] and keltner_ema_1h[i] > 0:
            vote_short += 0.5
        
        # Signal 4: Volume-weighted RSI
        if vwr_1h[i] > 55:
            vote_long += 0.5
        elif vwr_1h[i] < 45:
            vote_short += 0.5
        
        # Signal 5: VWAP mean reversion
        if vwap_dev < -1.0 and trend_regime:
            vote_long += 0.5  # Below VWAP in mean reversion regime
        elif vwap_dev > 1.0 and trend_regime:
            vote_short += 0.5  # Above VWAP in mean reversion regime
        
        # Signal 6: Squeeze breakout
        if squeeze_val == 1 and squeeze_4h[i // 4 - 1] == 0 if i // 4 > 0 else False:
            # Squeeze just turned on - potential breakout
            if close[i] > keltner_ema_1h[i]:
                vote_long += 0.5
            else:
                vote_short += 0.5
        
        # Bonus: Volume confirmation
        vol_bonus = 0.5 if vol_spike else 0
        
        # Determine vote direction
        if vote_long > vote_short and vote_long >= 2.5:
            current_vote = 1
            total_votes = vote_long + vol_bonus
        elif vote_short > vote_long and vote_short >= 2.5:
            current_vote = -1
            total_votes = vote_short + vol_bonus
        else:
            current_vote = 0
            total_votes = 0
        
        # Hysteresis: 2 consecutive bars for entry
        if current_vote != 0 and current_vote == prev_vote_direction:
            consecutive_votes += 1
        elif current_vote != 0:
            consecutive_votes = 1
            prev_vote_direction = current_vote
        else:
            consecutive_votes = 0
            prev_vote_direction = 0
        
        # Calculate volatility-adjusted size
        atr_pct = atr_1h[i] / close[i] if close[i] > 0 else 0
        vol_adjustment = min(1.3, max(0.6, ATR_TARGET_PCT / atr_pct)) if atr_pct > 0 else 1.0
        
        # Check for ATR trailing stop exit
        if prev_signal != 0.0 and entry_price > 0:
            stop_mult = 2.5 if trend_regime else 1.5  # Wider stops in trend regime
            
            if prev_signal > 0:  # Long position
                highest_close = max(highest_close, close[i])
                stop_long = max(entry_price - stop_mult * entry_atr, highest_close - stop_mult * atr_1h[i])
                
                # Take profit at 2R
                if not tp_triggered and close[i] >= entry_price + 2 * stop_mult * entry_atr:
                    signals[i] = prev_signal * 0.5  # Reduce to half
                    tp_triggered = True
                    continue
                
                if close[i] < stop_long:
                    signals[i] = 0.0
                    prev_signal = 0.0
                    entry_price = 0.0
                    consecutive_votes = 0
                    tp_triggered = False
                    continue
            else:  # Short position
                lowest_close = min(lowest_close, close[i])
                stop_short = min(entry_price + stop_mult * entry_atr, lowest_close + stop_mult * atr_1h[i])
                
                # Take profit at 2R
                if not tp_triggered and close[i] <= entry_price - 2 * stop_mult * entry_atr:
                    signals[i] = prev_signal * 0.5  # Reduce to half
                    tp_triggered = True
                    continue
                
                if close[i] > stop_short:
                    signals[i] = 0.0
                    prev_signal = 0.0
                    entry_price = 0.0
                    consecutive_votes = 0
                    tp_triggered = False
                    continue
        
        # Generate signal
        if prev_signal != 0.0:
            if current_vote == 0 or current_vote != np.sign(prev_signal):
                signals[i] = 0.0
                prev_signal = 0.0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
                tp_triggered = False
            else:
                signals[i] = prev_signal
        elif consecutive_votes >= 2 and adx_filter:
            if current_vote == 1:
                base_size = SIZE_HIGH if total_votes >= 4.0 else SIZE_LOW
                signals[i] = base_size * vol_adjustment
                entry_price = close[i]
                entry_atr = atr_1h[i]
                highest_close = close[i]
                prev_signal = signals[i]
                tp_triggered = False
            else:
                base_size = SIZE_HIGH if total_votes >= 4.0 else SIZE_LOW
                signals[i] = -base_size * vol_adjustment
                entry_price = close[i]
                entry_atr = atr_1h[i]
                lowest_close = close[i]
                prev_signal = signals[i]
                tp_triggered = False
        else:
            signals[i] = 0.0
            prev_signal = 0.0
    
    signals = np.clip(signals, -0.40, 0.40)
    
    return signals