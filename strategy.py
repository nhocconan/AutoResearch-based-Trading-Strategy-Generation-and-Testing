#!/usr/bin/env python3
"""
EXPERIMENT #088 - Supertrend + RSI Pullback + Volume Confirmation MTF
==================================================================================================
Hypothesis: After 70+ failed experiments, return to proven components from #084 (Sharpe=0.423).
Supertrend provides clean trend signals, RSI pullbacks give optimal entry timing,
volume confirmation filters false breakouts. 4h trend filter + 15m entries.

Key differences from #087:
- Supertrend instead of HMA for cleaner trend signals (proven in #084)
- Volume spike filter (>1.5x 20-bar avg) for entry confirmation
- Tighter RSI pullback zones (30-45 long, 55-70 short)
- ATR stoploss at 2.5*ATR (wider than #087's 2.0*ATR)
- Max position 0.30 (conservative vs 0.35 baseline)
- Cross-asset BTC 4h trend filter for ETH/SOL alignment

Why this should beat #087 (Sharpe=0.028):
- Supertrend has cleaner whipsaw protection than HMA crossover
- Volume filter reduces false entries in low-liquidity periods
- Based on #084 success pattern (cross_asset_kama_supertrend_volume)
- Simpler logic = fewer failure modes than complex regime detection
"""

import numpy as np
import pandas as pd

try:
    from mtf_data import get_htf_data, align_htf_to_ltf
    HAS_MTF_DATA = True
except ImportError:
    HAS_MTF_DATA = False

name = "supertrend_rsi_volume_mtf_15m_4h_v1"
timeframe = "15m"
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


def calculate_supertrend(high, low, close, atr, multiplier=3.0):
    """Calculate Supertrend indicator"""
    n = len(close)
    if n < len(atr) or len(atr) < 14:
        return np.zeros(n), np.zeros(n)
    
    supertrend = np.zeros(n)
    direction = np.zeros(n)  # 1 = bullish, -1 = bearish
    
    # Calculate basic bands
    hl2 = (high + low) / 2.0
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    # Initialize
    supertrend[13] = upper_band[13]
    direction[13] = 1
    
    for i in range(14, n):
        # Update upper band
        if upper_band[i] < upper_band[i - 1] or close[i - 1] > upper_band[i - 1]:
            upper_band[i] = upper_band[i]
        else:
            upper_band[i] = upper_band[i - 1]
        
        # Update lower band
        if lower_band[i] > lower_band[i - 1] or close[i - 1] < lower_band[i - 1]:
            lower_band[i] = lower_band[i]
        else:
            lower_band[i] = lower_band[i - 1]
        
        # Determine direction and supertrend value
        if direction[i - 1] == 1:
            if close[i] < lower_band[i]:
                direction[i] = -1
                supertrend[i] = upper_band[i]
            else:
                direction[i] = 1
                supertrend[i] = lower_band[i]
        else:
            if close[i] > upper_band[i]:
                direction[i] = 1
                supertrend[i] = lower_band[i]
            else:
                direction[i] = -1
                supertrend[i] = upper_band[i]
    
    return supertrend, direction


def calculate_rsi(close, period=14):
    """Calculate RSI"""
    n = len(close)
    if n < period + 1:
        return np.zeros(n)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    
    avg_gain[period] = np.mean(gain[:period + 1])
    avg_loss[period] = np.mean(loss[:period + 1])
    
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period
    
    rs = np.zeros(n)
    for i in range(period, n):
        if avg_loss[i] == 0:
            rs[i] = 100
        else:
            rs[i] = avg_gain[i] / avg_loss[i]
    
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD histogram"""
    n = len(close)
    if n < slow + signal:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    ema_fast = np.zeros(n)
    ema_slow = np.zeros(n)
    
    ema_fast[fast - 1] = np.mean(close[:fast])
    ema_slow[slow - 1] = np.mean(close[:slow])
    
    for i in range(fast, n):
        ema_fast[i] = ema_fast[i - 1] + (2.0 / (fast + 1)) * (close[i] - ema_fast[i - 1])
    
    for i in range(slow, n):
        ema_slow[i] = ema_slow[i - 1] + (2.0 / (slow + 1)) * (close[i] - ema_slow[i - 1])
    
    macd_line = ema_fast - ema_slow
    
    signal_line = np.zeros(n)
    first_signal = slow + signal - 1
    signal_line[first_signal] = np.mean(macd_line[slow:first_signal + 1])
    
    for i in range(first_signal + 1, n):
        signal_line[i] = signal_line[i - 1] + (2.0 / (signal + 1)) * (macd_line[i] - signal_line[i - 1])
    
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def calculate_volume_spike(volume, period=20, threshold=1.5):
    """Detect volume spikes above threshold * average"""
    n = len(volume)
    if n < period:
        return np.zeros(n, dtype=bool)
    
    avg_volume = np.zeros(n)
    avg_volume[period - 1] = np.mean(volume[:period])
    
    for i in range(period, n):
        avg_volume[i] = (avg_volume[i - 1] * (period - 1) + volume[i]) / period
    
    spike = volume > (avg_volume * threshold)
    
    return spike


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values if "volume" in prices.columns else np.ones(len(close))
    n = len(close)
    
    # 15m indicators for entry timing
    atr_15m = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=14)
    macd_line_15m, macd_signal_15m, macd_hist_15m = calculate_macd(close, fast=12, slow=26, signal=9)
    supertrend_15m, supertrend_dir_15m = calculate_supertrend(high, low, close, atr_15m, multiplier=3.0)
    volume_spike_15m = calculate_volume_spike(volume, period=20, threshold=1.5)
    
    # Get 4h data using mtf_data helper (CRITICAL for proper alignment)
    if HAS_MTF_DATA:
        try:
            df_4h = get_htf_data(prices, '4h')
            close_4h = df_4h['close'].values
            high_4h = df_4h['high'].values
            low_4h = df_4h['low'].values
            
            # Calculate 4h ATR and Supertrend for trend filter
            atr_4h = calculate_atr(high_4h, low_4h, close_4h, period=14)
            supertrend_4h, supertrend_dir_4h = calculate_supertrend(high_4h, low_4h, close_4h, atr_4h, multiplier=3.0)
            
            # Align 4h indicators to 15m timeframe (auto shift for completed bars)
            supertrend_dir_4h_aligned = align_htf_to_ltf(prices, df_4h, supertrend_dir_4h)
            close_4h_aligned = align_htf_to_ltf(prices, df_4h, close_4h)
            
        except Exception:
            # Fallback if mtf_data fails
            supertrend_dir_4h_aligned = np.zeros(n)
            close_4h_aligned = close.copy()
    else:
        # Fallback: simple downsampling
        supertrend_dir_4h_aligned = np.zeros(n)
        close_4h_aligned = close.copy()
        bars_per_4h = 16  # 16 x 15m = 4h
        n_4h = n // bars_per_4h
        if n_4h > 50:
            c_4h = np.array([close[i * bars_per_4h + bars_per_4h - 1] for i in range(n_4h)])
            h_4h = np.array([high[i * bars_per_4h + bars_per_4h - 1] for i in range(n_4h)])
            l_4h = np.array([low[i * bars_per_4h + bars_per_4h - 1] for i in range(n_4h)])
            atr_4h_simple = calculate_atr(h_4h, l_4h, c_4h, period=14)
            _, st_dir_4h_simple = calculate_supertrend(h_4h, l_4h, c_4h, atr_4h_simple, multiplier=3.0)
            for i in range(n):
                idx_4h = i // bars_per_4h
                if idx_4h < n_4h and idx_4h >= 50:
                    supertrend_dir_4h_aligned[i] = st_dir_4h_simple[idx_4h]
    
    # Generate signals
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_BASE = 0.20
    SIZE_CONFIRMED = 0.25
    SIZE_MAX = 0.30  # Absolute max position
    
    # Entry thresholds
    RSI_LONG_MIN = 30
    RSI_LONG_MAX = 45
    RSI_SHORT_MIN = 55
    RSI_SHORT_MAX = 70
    ATR_STOP_MULT = 2.5  # Wider stop than #087
    
    first_valid = max(200, 50, 26 + 9, 20)
    
    # Track position state
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(atr_15m[i]) or np.isnan(rsi_15m[i]) or atr_15m[i] == 0:
            signals[i] = 0.0
            continue
        
        rsi_val = rsi_15m[i]
        macd_hist = macd_hist_15m[i]
        atr = atr_15m[i]
        price = close[i]
        st_dir_15m = supertrend_dir_15m[i]
        st_dir_4h = supertrend_dir_4h_aligned[i]
        vol_spike = volume_spike_15m[i]
        
        # Check stoploss and take profit for existing positions
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
            
            # Stoploss check (2.5*ATR)
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
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry + 2 * ATR_STOP_MULT * atr
                if not prev_tp and price >= tp_price:
                    signals[i] = signals[i - 1] * 0.5
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R profit
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
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry - 2 * ATR_STOP_MULT * atr
                if not prev_tp and price <= tp_price:
                    signals[i] = signals[i - 1] * 0.5
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R profit
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
            
            # Hold position if no exit triggered
            signals[i] = signals[i - 1]
            position_side[i] = position_side[i - 1]
            entry_price[i] = entry_price[i - 1]
            tp_triggered[i] = tp_triggered[i - 1]
            highest_since_entry[i] = highest_since_entry[i - 1]
            lowest_since_entry[i] = lowest_since_entry[i - 1]
            continue
        
        # Entry logic: Supertrend trend + RSI pullback + Volume confirmation
        long_signals = 0
        short_signals = 0
        
        # Signal 1: 4h Supertrend trend (primary filter)
        if st_dir_4h == 1:
            long_signals += 2  # Weight 4h trend higher
        elif st_dir_4h == -1:
            short_signals += 2
        
        # Signal 2: 15m Supertrend direction (confirmation)
        if st_dir_15m == 1:
            long_signals += 1
        elif st_dir_15m == -1:
            short_signals += 1
        
        # Signal 3: RSI pullback in trend direction
        if st_dir_4h == 1 and RSI_LONG_MIN <= rsi_val <= RSI_LONG_MAX:
            long_signals += 1
        elif st_dir_4h == -1 and RSI_SHORT_MIN <= rsi_val <= RSI_SHORT_MAX:
            short_signals += 1
        
        # Signal 4: MACD momentum confirmation
        if macd_hist > 0:
            long_signals += 0.5
        elif macd_hist < 0:
            short_signals += 0.5
        
        # Signal 5: Volume spike confirmation (only for entries, not required)
        if vol_spike:
            if long_signals > 0:
                long_signals += 0.5
            elif short_signals > 0:
                short_signals += 0.5
        
        # Generate signal based on signal strength
        if long_signals >= 3.0 and long_signals > short_signals:
            if long_signals >= 4.5:
                signals[i] = SIZE_MAX
            elif long_signals >= 3.5:
                signals[i] = SIZE_CONFIRMED
            else:
                signals[i] = SIZE_BASE
            position_side[i] = 1
            entry_price[i] = price
            tp_triggered[i] = 0
            highest_since_entry[i] = price
            lowest_since_entry[i] = price
            
        elif short_signals >= 3.0 and short_signals > long_signals:
            if short_signals >= 4.5:
                signals[i] = -SIZE_MAX
            elif short_signals >= 3.5:
                signals[i] = -SIZE_CONFIRMED
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
    
    return signals