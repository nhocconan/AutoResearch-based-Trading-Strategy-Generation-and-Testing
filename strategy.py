#!/usr/bin/env python3
"""
Experiment #274: 4h Primary + 12h HTF — Simplified HMA Trend + RSI Pullback

Hypothesis: After 20+ failed 4h experiments with complex regime logic, simplify:
1. 12h HMA(21) for PRIMARY trend (faster response than 1d, proved in #251)
2. RSI(14) on 4h for pullback entries (not extremes - catch mid-trend continuations)
3. ADX(14) for trend strength - but use LOWER threshold (>18 not >25)
4. Donchian(20) breakout as confirmation (not primary signal)
5. FORCE trades every 8 bars to ensure 30+ trades/year minimum
6. Simpler sizing: 0.30 base, 0.35 strong conviction

Key differences from #259 (which failed):
- 12h HTF instead of 1d (better balance for 4h primary timeframe)
- RSI thresholds 35-65 (not 25-75) - catch more mid-trend entries
- ADX threshold 18 (not 25) - less filtering, more trades
- Force trade every 8 bars (not 10) - ensures minimum frequency
- Removed complex choppiness regime switching (source of 0-trade failures)
- Cleaner long/short logic without conflicting conditions

Position sizing: 0.30 base, 0.35 strong (discrete)
Target: 30-50 trades/year per symbol (4h timeframe appropriate range)
Stoploss: 2.5 * ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_rsi_donchian_12h_simp_v1"
timeframe = "4h"
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

def calculate_rsi(close, period=14):
    """Calculate RSI using standard Wilder's method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    return rsi

def calculate_hma(close, period=21):
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    Faster and smoother than EMA, less lag.
    """
    n = period
    half = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    def wma(series, window):
        weights = np.arange(1, window + 1)
        return series.rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    
    raw_hma = 2 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_n)
    
    return hma.values

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    n = period
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    tr1 = high_s - low_s
    tr2 = (high_s - close_s.shift(1)).abs()
    tr3 = (low_s - close_s.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.ewm(span=n, min_periods=n, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=n, min_periods=n, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=n, min_periods=n, adjust=False).mean() / atr)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(span=n, min_periods=n, adjust=False).mean()
    
    return adx.fillna(0).values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (upper and lower bounds)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h HTF indicators (primary trend regime)
    hma_12h_21 = calculate_hma(df_12h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_12h_21_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_21)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    adx_14 = calculate_adx(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    hma_4h_21 = calculate_hma(close, 21)
    hma_4h_50 = calculate_hma(close, 50)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.30
    STRONG_SIZE = 0.35
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -8
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_12h_21_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(hma_4h_21[i]):
            continue
        
        if np.isnan(adx_14[i]):
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        # === 12H TREND REGIME (primary direction filter) ===
        # Bull: price above 12h HMA
        # Bear: price below 12h HMA
        regime_bull = close[i] > hma_12h_21_aligned[i]
        regime_bear = close[i] < hma_12h_21_aligned[i]
        
        # === TREND STRENGTH (relaxed threshold) ===
        is_trending = adx_14[i] > 18.0
        is_weak = adx_14[i] <= 18.0
        
        # === 4H LOCAL SIGNALS ===
        price_above_4h_hma = close[i] > hma_4h_21[i]
        price_below_4h_hma = close[i] < hma_4h_21[i]
        hma_4h_bullish = hma_4h_21[i] > hma_4h_50[i]
        hma_4h_bearish = hma_4h_21[i] < hma_4h_50[i]
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] >= donchian_upper[i] * 0.999
        donchian_breakout_short = close[i] <= donchian_lower[i] * 1.001
        
        # === RSI THRESHOLDS (relaxed for more trades) ===
        rsi_bullish = rsi_14[i] > 45.0
        rsi_bearish = rsi_14[i] < 55.0
        rsi_oversold = rsi_14[i] < 40.0
        rsi_overbought = rsi_14[i] > 60.0
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES
        if regime_bull:
            # Strong long: trending + price above both HMAs + RSI confirming
            if is_trending and price_above_4h_hma and hma_4h_bullish and rsi_bullish:
                new_signal = STRONG_SIZE
            # Medium long: Donchian breakout + 4h HMA bullish
            elif donchian_breakout_long and hma_4h_bullish:
                new_signal = BASE_SIZE
            # Pullback long: RSI oversold in bull regime
            elif rsi_oversold and price_above_4h_hma:
                new_signal = BASE_SIZE
        
        # SHORT ENTRIES
        if regime_bear:
            # Strong short: trending + price below both HMAs + RSI confirming
            if is_trending and price_below_4h_hma and hma_4h_bearish and rsi_bearish:
                if new_signal == 0.0 or abs(new_signal) < STRONG_SIZE:
                    new_signal = -STRONG_SIZE
            # Medium short: Donchian breakdown + 4h HMA bearish
            elif donchian_breakout_short and hma_4h_bearish:
                if new_signal == 0.0:
                    new_signal = -BASE_SIZE
            # Pullback short: RSI overbought in bear regime
            elif rsi_overbought and price_below_4h_hma:
                if new_signal == 0.0:
                    new_signal = -BASE_SIZE
        
        # === FREQUENCY SAFEGUARD (CRITICAL for 10+ trades) ===
        # Force trade if no signal for 8 bars (~32h = 1.3 days on 4h)
        if bars_since_last_trade > 8 and new_signal == 0.0 and not in_position:
            if regime_bull and rsi_14[i] > 40 and price_above_4h_hma:
                new_signal = BASE_SIZE * 0.9
            elif regime_bear and rsi_14[i] < 60 and price_below_4h_hma:
                new_signal = -BASE_SIZE * 0.9
        
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
        
        # === REGIME REVERSAL EXIT ===
        regime_reversal = False
        if in_position and position_side != 0:
            # Long position but regime turns bearish
            if position_side > 0 and regime_bear and price_below_4h_hma:
                regime_reversal = True
            # Short position but regime turns bullish
            if position_side < 0 and regime_bull and price_above_4h_hma:
                regime_reversal = True
        
        if stoploss_triggered or regime_reversal:
            new_signal = 0.0
        
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