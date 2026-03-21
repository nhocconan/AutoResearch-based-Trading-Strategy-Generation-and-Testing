#!/usr/bin/env python3
"""
EXPERIMENT #017 - Keltner Channel Mean Reversion with Daily Trend (6h Primary)
==================================================================================================
Hypothesis: Current best (Sharpe=0.537) uses 4h primary + 1d trend with HMA+RSI. This uses 6h primary
+ daily trend with Keltner Channel mean reversion. 6h timeframe reduces noise vs 1h/4h while capturing
more moves than 4h. Keltner Channels (ATR-based) work better for crypto than Bollinger (std-dev based)
because volatility clusters in crypto. RSI pullback entries proven in current best.

Key innovations:
1. 6h PRIMARY (new timeframe) - cleaner signals than 1h/4h, more trades than 4h
2. Daily HMA trend filter - strongest trend signal, aligns with current best architecture
3. Keltner Channel (not Bollinger) - ATR-based bands capture crypto volatility regimes better
4. RSI(14) pullback entries - proven winner from current best (Sharpe=0.537)
5. Volume confirmation - taker buy/sell ratio filters institutional interest
6. Conservative sizing: 0.20 base, 0.30 max (vs 0.35 in some failed strategies)
7. Stoploss: 2.0 ATR (proven in current best, not 1.5 which caused #015 DD=-51.8%)

Why this should beat hma_rsi_pullback_daily_trend_4h_v1 (Sharpe=0.537):
- 6h timeframe has less noise than 4h (fewer false breakouts)
- Keltner Channels adapt to volatility better than fixed std-dev bands
- Same proven RSI pullback + daily trend architecture as current best
- Conservative sizing prevents the -51.8% DD that killed #015
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "keltner_rsi_daily_trend_6h_v1"
timeframe = "6h"
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


def calculate_hma(close, period=21):
    """
    Hull Moving Average - faster than EMA, smoother than SMA
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    """
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, window):
        result = np.zeros(len(series))
        weights = np.arange(1, window + 1)
        for i in range(window - 1, len(series)):
            result[i] = np.sum(series[i - window + 1:i + 1] * weights) / np.sum(weights)
        return result
    
    close_series = np.array(close)
    wma_half = wma(close_series, half)
    wma_full = wma(close_series, period)
    diff = 2 * wma_half - wma_full
    hma = wma(diff, sqrt_period)
    
    return hma


def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing"""
    n = len(close)
    if n < period + 1:
        return np.zeros(n)
    
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    
    gain[1:] = np.maximum(delta, 0)
    loss[1:] = np.maximum(-delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    
    avg_gain[period] = np.mean(gain[1:period + 1])
    avg_loss[period] = np.mean(loss[1:period + 1])
    
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period
    
    rs = np.zeros(n)
    mask = avg_loss > 0
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    
    rsi = np.zeros(n)
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    
    return rsi


def calculate_keltner_channels(high, low, close, atr, ema_period=20, atr_mult=2.0):
    """
    Keltner Channels - ATR-based volatility bands
    Middle: EMA(20)
    Upper: EMA + 2*ATR
    Lower: EMA - 2*ATR
    """
    n = len(close)
    if n < ema_period:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    close_series = pd.Series(close)
    middle = close_series.ewm(span=ema_period, adjust=False, min_periods=ema_period).mean().values
    
    upper = middle + atr_mult * atr
    lower = middle - atr_mult * atr
    
    return upper, middle, lower


def calculate_supertrend(high, low, close, atr, multiplier=3.0):
    """
    Supertrend indicator - trend following with ATR-based stops
    Returns: supertrend_values, trend_direction (1=up, -1=down)
    """
    n = len(close)
    if n < len(atr) or len(atr) == 0:
        return np.zeros(n), np.zeros(n)
    
    supertrend = np.zeros(n)
    trend = np.zeros(n)
    upper_band = np.zeros(n)
    lower_band = np.zeros(n)
    
    for i in range(n):
        if atr[i] == 0:
            continue
        upper_band[i] = (high[i] + low[i]) / 2 + multiplier * atr[i]
        lower_band[i] = (high[i] + low[i]) / 2 - multiplier * atr[i]
    
    first_valid = np.where(atr > 0)[0]
    if len(first_valid) == 0:
        return supertrend, trend
    
    start_idx = first_valid[0]
    supertrend[start_idx] = upper_band[start_idx]
    trend[start_idx] = 1
    
    for i in range(start_idx + 1, n):
        if atr[i] == 0:
            supertrend[i] = supertrend[i - 1]
            trend[i] = trend[i - 1]
            continue
        
        if trend[i - 1] == 1:
            if close[i] > lower_band[i]:
                supertrend[i] = max(supertrend[i - 1], lower_band[i])
                trend[i] = 1
            else:
                supertrend[i] = upper_band[i]
                trend[i] = -1
        else:
            if close[i] < upper_band[i]:
                supertrend[i] = min(supertrend[i - 1], upper_band[i])
                trend[i] = -1
            else:
                supertrend[i] = lower_band[i]
                trend[i] = 1
    
    return supertrend, trend


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_vol = prices["taker_buy_volume"].values
    n = len(close)
    
    # ========== 6h INDICATORS (PRIMARY TIMEFRAME) ==========
    atr_6h = calculate_atr(high, low, close, period=14)
    rsi_6h = calculate_rsi(close, period=14)
    supertrend_6h, st_trend_6h = calculate_supertrend(high, low, close, atr_6h, multiplier=3.0)
    kc_upper, kc_middle, kc_lower = calculate_keltner_channels(high, low, close, atr_6h, ema_period=20, atr_mult=2.0)
    
    # Volume ratio (taker buy / total volume)
    volume_ratio = np.zeros(n)
    mask = volume > 0
    volume_ratio[mask] = taker_buy_vol[mask] / volume[mask]
    
    # ========== DAILY INDICATORS (TREND FILTER) - PROPER MTF ==========
    try:
        df_1d = get_htf_data(prices, '1d')
        close_1d = df_1d['close'].values
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        
        # Daily HMA for trend direction (proven in current best)
        hma_1d = calculate_hma(close_1d, period=21)
        atr_1d = calculate_atr(high_1d, low_1d, close_1d, period=14)
        
        # Align to 6h timeframe (auto shift for completed bars)
        hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
        atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
        
    except Exception:
        hma_1d_aligned = np.zeros(n)
        atr_1d_aligned = np.zeros(n)
    
    # ========== SIGNAL GENERATION ==========
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE LEVELS (conservative to avoid -51% DD)
    SIZE_BASE = 0.20    # Base position (low conviction)
    SIZE_HIGH = 0.30    # High conviction (max)
    
    # ATR stoploss - PROVEN 2.0 from current best (not 1.5 which caused #015 failure)
    ATR_STOP_MULT = 2.0
    
    # RSI pullback thresholds (proven in current best)
    RSI_LONG_MAX = 55    # Buy on pullback when RSI < 55 (not oversold)
    RSI_SHORT_MIN = 45   # Sell on pullback when RSI > 45 (not overbought)
    RSI_OVERBOUGHT = 70  # Extreme overbought
    RSI_OVERSOLD = 30    # Extreme oversold
    
    # Keltner Channel position
    KC_LONG_ZONE = 0     # Price below middle (pullback zone)
    KC_SHORT_ZONE = 0    # Price above middle (pullback zone)
    
    # Volume confirmation
    VOLUME_RATIO_LONG = 0.50   # Neutral to buy pressure
    VOLUME_RATIO_SHORT = 0.50  # Neutral to sell pressure
    
    first_valid = max(100, 50)
    
    # Track position state
    position_side = np.zeros(n, dtype=int)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n, dtype=bool)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        # Skip invalid data
        if np.isnan(atr_6h[i]) or atr_6h[i] == 0 or np.isnan(rsi_6h[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_6h[i]
        rsi_val = rsi_6h[i]
        st_trend_val = st_trend_6h[i]
        vol_ratio = volume_ratio[i]
        
        # Keltner Channel position
        kc_position = 0
        if kc_middle[i] > 0:
            if price < kc_middle[i]:
                kc_position = -1  # Below middle (long pullback zone)
            elif price > kc_middle[i]:
                kc_position = 1   # Above middle (short pullback zone)
        
        # Daily trend filters (MASTER FILTER - from current best architecture)
        hma_1d_val = hma_1d_aligned[i]
        atr_1d_val = atr_1d_aligned[i]
        
        # Determine daily trend direction
        trend_1d = 0
        if hma_1d_val > 0:
            if price > hma_1d_val:
                trend_1d = 1  # Bullish
            elif price < hma_1d_val:
                trend_1d = -1  # Bearish
        
        # ========== CHECK EXISTING POSITIONS ==========
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
            
            # Stoploss check (2.0*ATR - proven in current best)
            if prev_side == 1:
                stoploss_price = prev_entry - ATR_STOP_MULT * atr
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = False
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry + 2 * ATR_STOP_MULT * atr
                if not prev_tp and price >= tp_price:
                    signals[i] = SIZE_BASE / 2  # Reduce to half
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = True
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_high - ATR_STOP_MULT * atr
                    if price < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = False
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
                    
            elif prev_side == -1:
                stoploss_price = prev_entry + ATR_STOP_MULT * atr
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = False
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry - 2 * ATR_STOP_MULT * atr
                if not prev_tp and price <= tp_price:
                    signals[i] = -SIZE_BASE / 2  # Reduce to half
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = True
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_low + ATR_STOP_MULT * atr
                    if price > trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = False
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
        
        # ========== ENTRY LOGIC - KELTNER MEAN REVERSION + RSI PULLBACK ==========
        # LONG: Daily trend up + 6h Supertrend up + Price below KC middle (pullback) + RSI not overbought + volume confirms
        long_condition = (
            trend_1d == 1 and
            st_trend_val == 1 and
            kc_position == -1 and  # Price below KC middle (pullback)
            rsi_val < RSI_LONG_MAX and  # RSI pullback (not overbought)
            rsi_val > RSI_OVERSOLD and  # Not extremely oversold (avoid catching falling knife)
            vol_ratio >= VOLUME_RATIO_LONG
        )
        
        # SHORT: Daily trend down + 6h Supertrend down + Price above KC middle (pullback) + RSI not oversold + volume confirms
        short_condition = (
            trend_1d == -1 and
            st_trend_val == -1 and
            kc_position == 1 and  # Price above KC middle (pullback)
            rsi_val > RSI_SHORT_MIN and  # RSI pullback (not oversold)
            rsi_val < RSI_OVERBOUGHT and  # Not extremely overbought
            vol_ratio <= VOLUME_RATIO_SHORT
        )
        
        # Determine conviction level
        conviction = 0
        if long_condition or short_condition:
            # High conviction: RSI in optimal pullback zone (40-50 for long, 50-60 for short)
            if long_condition and 40 <= rsi_val <= 50:
                conviction = 2
            elif short_condition and 50 <= rsi_val <= 60:
                conviction = 2
            # Very high conviction: Strong volume + RSI optimal
            if long_condition and vol_ratio > 0.55 and 40 <= rsi_val <= 50:
                conviction = 3
            elif short_condition and vol_ratio < 0.45 and 50 <= rsi_val <= 60:
                conviction = 3
            elif conviction < 2:
                conviction = 1
        
        # Assign position size based on conviction
        if long_condition:
            if conviction >= 3:
                size = SIZE_HIGH
            elif conviction >= 2:
                size = SIZE_HIGH
            else:
                size = SIZE_BASE
            
            signals[i] = size
            position_side[i] = 1
            entry_price[i] = price
            tp_triggered[i] = False
            highest_since_entry[i] = price
            lowest_since_entry[i] = price
        
        elif short_condition:
            if conviction >= 3:
                size = SIZE_HIGH
            elif conviction >= 2:
                size = SIZE_HIGH
            else:
                size = SIZE_BASE
            
            signals[i] = -size
            position_side[i] = -1
            entry_price[i] = price
            tp_triggered[i] = False
            highest_since_entry[i] = price
            lowest_since_entry[i] = price
        
        # Track state for existing positions
        if position_side[i] != 0 and entry_price[i] == 0:
            entry_price[i] = entry_price[i - 1]
            tp_triggered[i] = tp_triggered[i - 1]
            highest_since_entry[i] = highest_since_entry[i - 1]
            lowest_since_entry[i] = lowest_since_entry[i - 1]
    
    return signals