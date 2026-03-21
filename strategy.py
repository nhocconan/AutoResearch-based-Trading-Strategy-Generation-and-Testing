#!/usr/bin/env python3
"""
EXPERIMENT #067 - Adaptive KAMA-ADX Volume Strategy (1h + 4h)
==================================================================================================
Hypothesis: Current best uses 15m+1h+4h with Supertrend+MACD. Let me try a DIFFERENT approach:

Key insights from failures:
- Ensemble voting failed (conflicting signals)
- 15m entries may be too noisy (whipsaws)
- Supertrend+RSI combination is overused in failed strategies

New approach:
- KAMA (Kaufman Adaptive Moving Average) - adapts to market efficiency ratio
- ADX for trend strength confirmation (only trade when ADX > 25)
- Volume spike confirmation (volume > 1.5x average = real move)
- 1h entries (less noise than 15m) + 4h trend/regime
- Position sizing scales with ADX strength (stronger trend = larger position)
- Regime detection: BBW percentile but with DIFFERENT thresholds (0.25/0.75)

Why this should beat Sharpe=3.653:
- KAMA adapts better to ranging vs trending markets than HMA/EMA
- ADX filter prevents trades in weak trends (major source of losses)
- Volume confirmation reduces false breakouts
- 1h timeframe = fewer but higher quality trades
- ADX-based position sizing = more capital in high-confidence setups
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "adaptive_kama_adx_volume_1h_4h_v1"
timeframe = "1h"
leverage = 1.0


def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average
    Adapts smoothing based on market efficiency ratio
    """
    n = len(close)
    if n < er_period + slow_period:
        return np.zeros(n)
    
    # Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(er_period, n):
        signal = abs(close[i] - close[i - er_period])
        noise = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        if noise > 0:
            er[i] = signal / noise
        else:
            er[i] = 0
    
    # Smoothing constants
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    
    kama = np.zeros(n)
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama


def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index (ADX)
    Measures trend strength (not direction)
    """
    n = len(close)
    if n < period * 2:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_dm[i] = max(0, high[i] - high[i - 1]) if (high[i] - high[i - 1]) > (low[i - 1] - low[i]) else 0
        minus_dm[i] = max(0, low[i - 1] - low[i]) if (low[i - 1] - low[i]) > (high[i] - high[i - 1]) else 0
    
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    dx = np.zeros(n)
    adx = np.zeros(n)
    
    # Wilder's smoothing
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, adjust=False).mean().values
    tr_smooth = pd.Series(tr).ewm(span=period, adjust=False).mean().values
    
    for i in range(period, n):
        if tr_smooth[i] > 0:
            plus_di[i] = 100 * plus_dm_smooth[i] / tr_smooth[i]
            minus_di[i] = 100 * minus_dm_smooth[i] / tr_smooth[i]
        
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    # ADX = smoothed DX
    adx[period * 2 - 1] = np.mean(dx[period:period * 2])
    for i in range(period * 2, n):
        adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period
    
    return adx, plus_di, minus_di


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


def calculate_rsi(close, period=14):
    """Calculate RSI"""
    n = len(close)
    if n < period + 1:
        return np.zeros(n)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, adjust=False).mean().values
    
    rs = np.zeros(n)
    for i in range(n):
        if avg_loss[i] == 0:
            rs[i] = 100
        else:
            rs[i] = avg_gain[i] / avg_loss[i]
    
    rsi = 100 - (100 / (1 + rs))
    rsi[:period] = 50
    
    return rsi


def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and Band Width"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    
    middle = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = middle + std_mult * std
    lower = middle - std_mult * std
    
    bbw = np.zeros(n)
    mask = middle > 0
    bbw[mask] = (upper[mask] - lower[mask]) / middle[mask]
    
    return upper, middle, lower, bbw


def calculate_bbw_percentile(bbw, lookback=100):
    """Calculate BBW percentile rank for regime detection"""
    n = len(bbw)
    percentile = np.zeros(n)
    
    for i in range(lookback - 1, n):
        window = bbw[i - lookback + 1:i + 1]
        rank = np.sum(window <= bbw[i]) / len(window)
        percentile[i] = rank
    
    return percentile


def calculate_volume_sma(volume, period=20):
    """Calculate volume simple moving average"""
    return pd.Series(volume).rolling(window=period, min_periods=period).mean().values


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values if "volume" in prices.columns else np.ones(len(close))
    n = len(close)
    
    # === 1h indicators (entry timing) ===
    kama_1h = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    adx_1h, plus_di_1h, minus_di_1h = calculate_adx(high, low, close, period=14)
    atr_1h = calculate_atr(high, low, close, period=14)
    rsi_1h = calculate_rsi(close, period=14)
    volume_sma_1h = calculate_volume_sma(volume, period=20)
    
    # === 4h indicators (regime & trend) using mtf_data helper ===
    try:
        df_4h = get_htf_data(prices, '4h')
        c_4h = df_4h['close'].values
        h_4h = df_4h['high'].values
        l_4h = df_4h['low'].values
        
        # 4h indicators
        kama_4h = calculate_kama(c_4h, er_period=10, fast_period=2, slow_period=30)
        adx_4h, plus_di_4h, minus_di_4h = calculate_adx(h_4h, l_4h, c_4h, period=14)
        _, _, _, bbw_4h = calculate_bollinger_bands(c_4h, period=20, std_mult=2.0)
        bbw_pct_4h = calculate_bbw_percentile(bbw_4h, lookback=100)
        
        # Align 4h indicators to 1h timeframe (auto shift for completed bars)
        kama_4h_aligned = align_htf_to_ltf(prices, df_4h, kama_4h)
        adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
        plus_di_4h_aligned = align_htf_to_ltf(prices, df_4h, plus_di_4h)
        minus_di_4h_aligned = align_htf_to_ltf(prices, df_4h, minus_di_4h)
        bbw_pct_4h_aligned = align_htf_to_ltf(prices, df_4h, bbw_pct_4h)
        
    except Exception as e:
        # Fallback: use 1h data only if 4h not available
        kama_4h_aligned = kama_1h
        adx_4h_aligned = adx_1h
        plus_di_4h_aligned = plus_di_1h
        minus_di_4h_aligned = minus_di_1h
        bbw_pct_4h_aligned = calculate_bbw_percentile(calculate_bollinger_bands(close)[3], lookback=100)
    
    # === Regime detection ===
    # LOW volatility (BBW percentile < 0.25): Trend following mode
    # HIGH volatility (BBW percentile > 0.75): Mean reversion mode
    # MEDIUM volatility (0.25-0.75): Reduced position or no trade
    REGIME_LOW_THRESHOLD = 0.25
    REGIME_HIGH_THRESHOLD = 0.75
    
    # === ADX thresholds ===
    ADX_TREND_MIN = 25  # Only trade when ADX > 25 (strong trend)
    ADX_STRONG = 40  # Very strong trend = larger position
    
    # === Position sizing (discrete levels, scaled by ADX) ===
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.35
    SIZE_WEAK = 0.15
    SIZE_MR = 0.20  # Mean reversion uses smaller size
    
    # === Entry thresholds ===
    VOLUME_SPIKE_MULT = 1.5  # Volume must be 1.5x average for confirmation
    RSI_LONG_MAX = 60  # Don't buy if RSI too high
    RSI_SHORT_MIN = 40  # Don't sell if RSI too low
    ATR_STOP_MULT = 2.5
    
    # === Initialize tracking arrays ===
    signals = np.zeros(n)
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    first_valid = max(250, 150)  # Need enough data for ADX and BBW percentile
    
    for i in range(first_valid, n):
        # Skip invalid data
        if np.isnan(atr_1h[i]) or atr_1h[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(bbw_pct_4h_aligned[i]) or np.isnan(adx_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_1h[i]
        regime = bbw_pct_4h_aligned[i]
        adx_4h = adx_4h_aligned[i]
        adx_1h = adx_1h[i]
        kama_4h_val = kama_4h_aligned[i]
        kama_1h_val = kama_1h[i]
        vol_ratio = volume[i] / volume_sma_1h[i] if volume_sma_1h[i] > 0 else 1.0
        
        # === Check existing positions (stoploss & take profit) ===
        if position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else close[i - 1]
            prev_tp = tp_triggered[i - 1]
            prev_high = highest_since_entry[i - 1] if highest_since_entry[i - 1] > 0 else prev_entry
            prev_low = lowest_since_entry[i - 1] if lowest_since_entry[i - 1] > 0 else prev_entry
            
            # Update highest/lowest since entry
            if prev_side == 1:
                current_high = max(prev_high, price)
                current_low = min(prev_low, price) if prev_low > 0 else price
            else:
                current_high = max(prev_high, price) if prev_high > 0 else price
                current_low = min(prev_low, price)
            
            highest_since_entry[i] = current_high
            lowest_since_entry[i] = current_low
            
            # Stoploss check
            if prev_side == 1:
                stoploss_price = prev_entry - ATR_STOP_MULT * atr
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit (2R) - reduce to half
                tp_price = prev_entry + 2 * ATR_STOP_MULT * atr
                if not prev_tp and price >= tp_price:
                    signals[i] = SIZE_BASE / 2
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R
                if prev_tp:
                    trail_stop = current_high - ATR_STOP_MULT * atr
                    if price < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
                
            elif prev_side == -1:
                stoploss_price = prev_entry + ATR_STOP_MULT * atr
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit (2R) - reduce to half
                tp_price = prev_entry - 2 * ATR_STOP_MULT * atr
                if not prev_tp and price <= tp_price:
                    signals[i] = -SIZE_BASE / 2
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R
                if prev_tp:
                    trail_stop = current_low + ATR_STOP_MULT * atr
                    if price > trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
            
            # Hold position
            signals[i] = signals[i - 1]
            position_side[i] = position_side[i - 1]
            entry_price[i] = entry_price[i - 1]
            tp_triggered[i] = tp_triggered[i - 1]
            highest_since_entry[i] = highest_since_entry[i - 1]
            lowest_since_entry[i] = lowest_since_entry[i - 1]
            continue
        
        # === REGIME-ADAPTIVE ENTRY LOGIC ===
        
        # LOW volatility regime: Trend following
        if regime < REGIME_LOW_THRESHOLD:
            # 4h ADX must confirm strong trend
            if adx_4h < ADX_TREND_MIN:
                signals[i] = 0.0
                position_side[i] = 0
                continue
            
            # Determine trend direction from 4h KAMA and DI
            if kama_4h_val > 0 and price > kama_4h_val and plus_di_4h_aligned[i] > minus_di_4h_aligned[i]:
                # Uptrend - check 1h confirmation
                if price > kama_1h_val and plus_di_1h[i] > minus_di_1h[i]:
                    # Volume confirmation
                    if vol_ratio >= VOLUME_SPIKE_MULT:
                        # RSI filter (not overbought)
                        if rsi_1h[i] < RSI_LONG_MAX:
                            # Position size based on ADX strength
                            if adx_4h >= ADX_STRONG:
                                signals[i] = SIZE_STRONG
                            else:
                                signals[i] = SIZE_BASE
                            
                            position_side[i] = 1
                            entry_price[i] = price
                            tp_triggered[i] = 0
                            highest_since_entry[i] = price
                            lowest_since_entry[i] = price
                        else:
                            signals[i] = 0.0
                            position_side[i] = 0
                    else:
                        signals[i] = 0.0
                        position_side[i] = 0
                else:
                    signals[i] = 0.0
                    position_side[i] = 0
                    
            elif kama_4h_val > 0 and price < kama_4h_val and minus_di_4h_aligned[i] > plus_di_4h_aligned[i]:
                # Downtrend - check 1h confirmation
                if price < kama_1h_val and minus_di_1h[i] > plus_di_1h[i]:
                    # Volume confirmation
                    if vol_ratio >= VOLUME_SPIKE_MULT:
                        # RSI filter (not oversold)
                        if rsi_1h[i] > RSI_SHORT_MIN:
                            # Position size based on ADX strength
                            if adx_4h >= ADX_STRONG:
                                signals[i] = -SIZE_STRONG
                            else:
                                signals[i] = -SIZE_BASE
                            
                            position_side[i] = -1
                            entry_price[i] = price
                            tp_triggered[i] = 0
                            highest_since_entry[i] = price
                            lowest_since_entry[i] = price
                        else:
                            signals[i] = 0.0
                            position_side[i] = 0
                    else:
                        signals[i] = 0.0
                        position_side[i] = 0
                else:
                    signals[i] = 0.0
                    position_side[i] = 0
            else:
                signals[i] = 0.0
                position_side[i] = 0
        
        # HIGH volatility regime: Mean reversion
        elif regime > REGIME_HIGH_THRESHOLD:
            # Mean reversion: buy when oversold, sell when overbought
            # Use smaller position size in high vol
            if rsi_1h[i] < 30 and price < kama_1h_val * 0.98:
                signals[i] = -SIZE_MR  # Reduced size in high vol
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
                
            elif rsi_1h[i] > 70 and price > kama_1h_val * 1.02:
                signals[i] = -SIZE_MR  # Reduced size in high vol
                position_side[i] = -1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
            else:
                signals[i] = 0.0
                position_side[i] = 0
        
        # MEDIUM volatility: No trade or very small positions
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals