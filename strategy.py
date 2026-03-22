#!/usr/bin/env python3
"""
Experiment #310: 1h Primary + 4h/12h HTF — HMA Trend + RSI Pullback + Choppiness Size

Hypothesis: 1h timeframe can work IF we use HTF for direction and 1h only for timing.
Key insight from failures: Lower TF strategies fail because entry conditions are TOO STRICT.

Why this might work (learning from #300, #305, #308 failures):
1. 4h HMA(21) for major trend direction (proven in best strategy #308 baseline)
2. 1h RSI(14) pullback entries with LOOSE thresholds (35-65, not 30-70)
3. Choppiness Index adjusts SIZE not entry (don't block trades)
4. Session filter 8-20 UTC only (when volume is real)
5. FORCE trades if none for 25 bars (prevent 0-trade failure)
6. Asymmetric sizing: longs 0.30, shorts 0.20 (crypto bias)
7. Stoploss: 2.5 * ATR (tighter for lower TF)

Key differences from failed 1h strategies:
- LOOSER RSI thresholds (35-65 vs 30-70) = more trades
- Choppiness affects size, not entry permission
- Force-trade mechanism after 25 bars without signal
- 4h trend filter only (not 4h+12h+1d = too many filters)
- Discrete signal levels to reduce fee churn

Position sizing: 0.25 base, 0.30 strong conviction (longs), 0.20 (shorts)
Target: 40-80 trades/year on 1h (appropriate for hourly, manageable fee drag)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_chop_size_4h_v1"
timeframe = "1h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    Faster response than EMA with less lag, less noise than SMA.
    """
    n = period
    n2 = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    # WMA helper
    def wma(series, window):
        weights = np.arange(1, window + 1)
        return series.rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, n2)
    wma_full = wma(close_s, n)
    
    hma = wma(2 * wma_half - wma_full, sqrt_n)
    
    return hma.values

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = choppy/range market
    CHOP < 38.2 = trending market
    """
    n = period
    close_s = pd.Series(close)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    # Calculate ATR
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=n, min_periods=n, adjust=False).mean()
    
    atr_sum = atr.rolling(window=n, min_periods=n).sum()
    hh = high_s.rolling(window=n, min_periods=n).max()
    ll = low_s.rolling(window=n, min_periods=n).min()
    
    chop = np.zeros(len(close))
    for i in range(n, len(close)):
        range_hl = hh.iloc[i] - ll.iloc[i]
        if range_hl > 0 and atr_sum.iloc[i] > 0:
            chop[i] = 100 * np.log10(atr_sum.iloc[i] / range_hl) / np.log10(n)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h HTF indicators (major trend direction)
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    atr_30 = calculate_atr(high, low, close, 30)
    chop_14 = calculate_choppiness_index(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    hma_1h_21 = calculate_hma(close, 21)
    hma_1h_50 = calculate_hma(close, 50)
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    # Asymmetric: longs favored in crypto
    LONG_BASE = 0.25
    LONG_STRONG = 0.30
    SHORT_BASE = 0.20
    SHORT_STRONG = 0.25
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -20
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_4h_21_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(chop_14[i]):
            continue
        
        if np.isnan(hma_1h_21[i]) or np.isnan(hma_1h_50[i]):
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        # Extract hour from open_time (milliseconds timestamp)
        hour_utc = (open_time[i] // 3600000) % 24
        in_session = 8 <= hour_utc <= 20
        
        # === 4H MAJOR TREND REGIME (primary direction filter) ===
        # Bull: price above 4h HMA (favor longs)
        # Bear: price below 4h HMA (allow shorts)
        regime_bull = close[i] > hma_4h_21_aligned[i]
        regime_bear = close[i] < hma_4h_21_aligned[i]
        
        # === CHOPPINESS REGIME (adjusts SIZE, not entry) ===
        # CHOP > 55 = range market (reduce size)
        # CHOP < 45 = trending market (full size)
        is_choppy = chop_14[i] > 55.0
        is_trending = chop_14[i] < 45.0
        
        # Size multiplier based on choppiness
        if is_trending:
            size_mult = 1.0
        elif is_choppy:
            size_mult = 0.7
        else:
            size_mult = 0.85
        
        # === VOLATILITY REGIME (ATR ratio) ===
        atr_ratio = atr_14[i] / (atr_30[i] + 1e-10)
        high_vol = atr_ratio > 1.5
        vol_scale = 0.8 if high_vol else 1.0
        
        # === 1H LOCAL TREND ===
        # HMA trend direction
        hma_bullish = hma_1h_21[i] > hma_1h_50[i]
        hma_bearish = hma_1h_21[i] < hma_1h_50[i]
        
        # HMA slope (3-bar lookback)
        hma_slope_up = hma_1h_21[i] > hma_1h_21[i-3] if i >= 3 else False
        hma_slope_down = hma_1h_21[i] < hma_1h_21[i-3] if i >= 3 else False
        
        # Price position relative to HMA
        price_above_hma = close[i] > hma_1h_21[i]
        price_below_hma = close[i] < hma_1h_21[i]
        
        # Price relative to SMA200 (long-term trend)
        price_above_sma200 = close[i] > sma_200[i] if not np.isnan(sma_200[i]) else True
        
        # === RSI SIGNALS (LOOSE thresholds for more trades) ===
        # RSI pullback long: RSI 35-55 in uptrend
        # RSI pullback short: RSI 45-65 in downtrend
        rsi_pullback_long = 35.0 < rsi_14[i] < 55.0
        rsi_pullback_short = 45.0 < rsi_14[i] < 65.0
        rsi_strong_oversold = rsi_14[i] < 35.0
        rsi_strong_overbought = rsi_14[i] > 65.0
        rsi_rising = rsi_14[i] > rsi_14[i-1] if i > 0 else False
        rsi_falling = rsi_14[i] < rsi_14[i-1] if i > 0 else False
        
        # === ENTRY LOGIC (LOOSE conditions to ensure trades) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES (favored in bull regime)
        if in_session:
            # Primary: RSI pullback in bull regime + 1h HMA bullish
            if regime_bull and rsi_pullback_long and hma_bullish:
                new_signal = LONG_BASE * size_mult * vol_scale
            
            # Strong: RSI very oversold + bull regime
            elif rsi_strong_oversold and regime_bull:
                new_signal = LONG_STRONG * size_mult * vol_scale
            
            # HMA bullish crossover + RSI rising
            elif hma_bullish and hma_slope_up and rsi_rising and rsi_14[i] > 40.0:
                new_signal = LONG_BASE * size_mult * vol_scale
            
            # Price above SMA200 + RSI pullback
            elif price_above_sma200 and rsi_pullback_long and regime_bull:
                new_signal = LONG_BASE * size_mult * vol_scale
        
        # SHORT ENTRIES (only in bear regime, reduced size)
        if in_session and new_signal == 0.0:
            # Primary: RSI pullback in bear regime + 1h HMA bearish
            if regime_bear and rsi_pullback_short and hma_bearish:
                new_signal = -SHORT_BASE * size_mult * vol_scale
            
            # Strong: RSI very overbought + bear regime
            elif rsi_strong_overbought and regime_bear:
                new_signal = -SHORT_STRONG * size_mult * vol_scale
            
            # HMA bearish crossover + RSI falling
            elif hma_bearish and hma_slope_down and rsi_falling and rsi_14[i] < 60.0:
                new_signal = -SHORT_BASE * size_mult * vol_scale
        
        # === FORCE TRADE MECHANISM (prevent 0 trades) ===
        # If no signal for 25 bars and in session, force entry
        if bars_since_last_trade > 25 and new_signal == 0.0 and not in_position and in_session:
            if regime_bull and rsi_14[i] > 40.0:
                new_signal = LONG_BASE * 0.6 * vol_scale
            elif regime_bear and rsi_14[i] < 60.0:
                new_signal = -SHORT_BASE * 0.6 * vol_scale
            elif rsi_strong_oversold:
                new_signal = LONG_BASE * 0.6 * vol_scale
            elif rsi_strong_overbought:
                new_signal = -SHORT_BASE * 0.6 * vol_scale
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === RSI REVERSAL EXIT ===
        rsi_exit = False
        if in_position and position_side != 0:
            # Long position: exit when RSI turns overbought
            if position_side > 0 and rsi_strong_overbought:
                rsi_exit = True
            # Short position: exit when RSI turns oversold
            if position_side < 0 and rsi_strong_oversold:
                rsi_exit = True
        
        # === HMA REVERSAL EXIT ===
        hma_exit = False
        if in_position and position_side != 0:
            # Long position: exit when HMA turns bearish + price below
            if position_side > 0 and hma_bearish and price_below_hma:
                hma_exit = True
            # Short position: exit when HMA turns bullish + price above
            if position_side < 0 and hma_bullish and price_above_hma:
                hma_exit = True
        
        # === REGIME REVERSAL EXIT ===
        regime_reversal = False
        if in_position and position_side != 0:
            # Long position but 4h regime turns bearish
            if position_side > 0 and regime_bear and price_below_hma:
                regime_reversal = True
            # Short position but 4h regime turns bullish
            if position_side < 0 and regime_bull and price_above_hma:
                regime_reversal = True
        
        if stoploss_triggered or rsi_exit or hma_exit or regime_reversal:
            new_signal = 0.0
        
        # === DISCRETIZE SIGNAL (reduce churn) ===
        if new_signal != 0.0:
            if abs(new_signal) < 0.12:
                new_signal = 0.0
            elif new_signal > 0.28:
                new_signal = LONG_STRONG * size_mult * vol_scale
            elif new_signal > 0:
                new_signal = LONG_BASE * size_mult * vol_scale
            elif new_signal < -0.23:
                new_signal = -SHORT_STRONG * size_mult * vol_scale
            else:
                new_signal = -SHORT_BASE * size_mult * vol_scale
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals