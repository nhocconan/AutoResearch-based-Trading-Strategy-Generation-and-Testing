#!/usr/bin/env python3
"""
Experiment #1108: 30m Primary + 4h/1d HTF — Simplified Trend-Follow with Loose Entries

Hypothesis: After 800+ failed experiments, the #1 lesson for 30m timeframe:
1. Session filters (8-20 UTC) KILL trade generation — REMOVE entirely
2. Volume filters are secondary — make them LOOSE or optional
3. Use 4h HMA for MACRO DIRECTION only (not entry timing)
4. Use 30m RSI for ENTRY TRIGGER with LOOSE thresholds (35/65 not 40/60)
5. Add Choppiness Index as REGIME FILTER but don't require perfect alignment
6. CRITICAL: At least ONE entry path must work independently (OR logic not AND)

Why this should beat 0-trade failures (#1098, #1105):
- No session filter = trades can happen 24/7
- Loose RSI (35/65) = more entry opportunities
- 4h HMA provides clean trend direction without over-complication
- 30m entries within 4h trend = HTF frequency with LTF precision
- Target: 40-80 trades/year (not 200+ which kills with fees)

Timeframe: 30m (primary)
HTF: 4h (trend), 1d (macro) — loaded ONCE before loop using mtf_data helper
Position Size: 0.25 base, 0.15 reduced (discrete levels)
Stoploss: 2.5x ATR trailing
Target: 40-80 trades/year, Sharpe > 0.612 (current best)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_hma_rsi_chop_4h1d_loose_v1"
timeframe = "30m"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average — reduces lag while maintaining smoothness.
    Formula: HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    def wma(data, span):
        """Weighted Moving Average."""
        if span < 1:
            span = 1
        result = np.full(len(data), np.nan)
        weights = np.arange(1, span + 1, dtype=float)
        for i in range(span - 1, len(data)):
            window = data[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    half = max(1, int(period / 2))
    wma1 = wma(close, half)
    wma2 = wma(close, period)
    
    diff = 2 * wma1 - wma2
    sqrt_period = max(1, int(np.sqrt(period)))
    
    hma = wma(diff, sqrt_period)
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index — momentum oscillator."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    diff = np.diff(close)
    gain = np.where(diff > 0, diff, 0.0)
    loss = np.where(diff < 0, -diff, 0.0)
    
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = avg_loss > 1e-10
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100.0 - (100.0 / (1.0 + rs[mask]))
    rsi[~mask] = 50.0
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility measurement."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index — measures market choppiness vs trending.
    CHOP > 61.8 = choppy/range, CHOP < 38.2 = trending
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    # Calculate ATR for each bar (simplified: just True Range)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    for i in range(period, n):
        tr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        price_range = highest_high - lowest_low
        if price_range > 1e-10 and tr_sum > 0:
            chop[i] = 100.0 * np.log10(tr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_ema(close, period=21):
    """Exponential Moving Average."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values if "volume" in prices.columns else np.ones(len(close))
    n = len(close)
    
    # === LOAD HTF DATA ONCE BEFORE LOOP (CRITICAL RULE #1) ===
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 4h HMA for trend direction
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d HMA for macro filter
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # === CALCULATE PRIMARY (30m) INDICATORS ===
    rsi_30m = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    ema_30m = calculate_ema(close, period=21)
    
    # Volume SMA for loose volume filter
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25
    REDUCED_SIZE = 0.15
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(rsi_30m[i]) or np.isnan(atr[i]) or np.isnan(chop[i]):
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(ema_30m[i]) or np.isnan(vol_sma[i]):
            continue
        if atr[i] <= 1e-10:
            continue
        
        # === MACRO TREND (1d HMA) — Only for bias, not hard filter ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === TREND DIRECTION (4h HMA) — Primary filter ===
        trend_bull = close[i] > hma_4h_aligned[i]
        trend_bear = close[i] < hma_4h_aligned[i]
        
        # === REGIME (Choppiness) — Loose filter ===
        # CHOP > 55 = range (prefer mean reversion)
        # CHOP < 45 = trend (prefer trend follow)
        # CHOP 45-55 = neutral (either works)
        is_choppy = chop[i] > 55.0
        is_trending = chop[i] < 45.0
        
        # === ENTRY TRIGGER (30m RSI) — LOOSE thresholds ===
        rsi_oversold = rsi_30m[i] < 35.0
        rsi_overbought = rsi_30m[i] > 65.0
        
        # === PRICE POSITION vs EMA ===
        price_above_ema = close[i] > ema_30m[i]
        price_below_ema = close[i] < ema_30m[i]
        
        # === VOLUME (LOOSE) — Just avoid dead periods ===
        volume_ok = volume[i] > 0.5 * vol_sma[i] if vol_sma[i] > 0 else True
        
        desired_signal = 0.0
        current_size = BASE_SIZE
        
        # === LONG ENTRY — Multiple paths (OR logic for trade generation) ===
        # Path 1: Trend follow (4h bull + 30m RSI pullback + price above EMA)
        long_trend = trend_bull and rsi_oversold and price_above_ema
        
        # Path 2: Macro + Trend alignment (1d bull + 4h bull + RSI < 50)
        long_macro = macro_bull and trend_bull and rsi_30m[i] < 50.0
        
        # Path 3: Range mean reversion (choppy + RSI very oversold)
        long_range = is_choppy and rsi_30m[i] < 30.0
        
        # Entry if ANY path triggers (OR logic ensures trades)
        if (long_trend or long_macro or long_range) and volume_ok:
            desired_signal = current_size
        
        # === SHORT ENTRY — Multiple paths (OR logic) ===
        # Path 1: Trend follow (4h bear + 30m RSI rally + price below EMA)
        short_trend = trend_bear and rsi_overbought and price_below_ema
        
        # Path 2: Macro + Trend alignment (1d bear + 4h bear + RSI > 50)
        short_macro = macro_bear and trend_bear and rsi_30m[i] > 50.0
        
        # Path 3: Range mean reversion (choppy + RSI very overbought)
        short_range = is_choppy and rsi_30m[i] > 70.0
        
        # Entry if ANY path triggers (OR logic ensures trades)
        if (short_trend or short_macro or short_range) and volume_ok:
            desired_signal = -current_size
        
        # === CONFLICT RESOLUTION — If both long and short signals ===
        # Prefer the direction aligned with 4h trend
        if trend_bull and desired_signal < 0:
            desired_signal = 0.0  # Cancel short in bull trend
        if trend_bear and desired_signal > 0:
            desired_signal = 0.0  # Cancel long in bear trend
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if trend intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if 4h trend still bull
                if trend_bull:
                    desired_signal = current_size
            elif position_side < 0:
                # Hold short if 4h trend still bear
                if trend_bear:
                    desired_signal = -current_size
        
        # === EXIT CONDITIONS — Take profit on RSI extremes ===
        if in_position and position_side > 0:
            # Exit long if 4h trend reverses or RSI very overbought
            if trend_bear or rsi_30m[i] > 75.0:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if 4h trend reverses or RSI very oversold
            if trend_bull or rsi_30m[i] < 25.0:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            if desired_signal >= BASE_SIZE * 0.8:
                desired_signal = BASE_SIZE
            elif desired_signal >= REDUCED_SIZE * 0.8:
                desired_signal = REDUCED_SIZE
            else:
                desired_signal = REDUCED_SIZE * 0.5
        elif desired_signal < 0:
            if desired_signal <= -BASE_SIZE * 0.8:
                desired_signal = -BASE_SIZE
            elif desired_signal <= -REDUCED_SIZE * 0.8:
                desired_signal = -REDUCED_SIZE
            else:
                desired_signal = -REDUCED_SIZE * 0.5
        else:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Flip position
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals