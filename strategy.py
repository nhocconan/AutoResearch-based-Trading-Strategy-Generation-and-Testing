#!/usr/bin/env python3
"""
Experiment #132: 12h Primary + 1d/1w HTF — Adaptive KAMA Trend + RSI Pullback + Choppiness Regime

Hypothesis: Recent failures stem from overly complex filters that prevent trade generation.
This strategy simplifies entry logic while maintaining edge through:

1. KAMA (Kaufman Adaptive Moving Average): Adapts to volatility - fast in trends, slow in chop
2. 1d HMA(21) Trend Bias: Major direction filter (long only when 1d HMA sloping up)
3. Choppiness Index Regime: CHOP>55 = mean revert (RSI extremes), CHOP<45 = trend follow
4. RSI(14) Pullback Entries: RSI<40 in uptrend for longs, RSI>60 in downtrend for shorts
5. ATR(14) Trailing Stop: 2.5*ATR protects capital during reversals

Why this should work:
- KAMA reduces whipsaw in choppy markets (2022, 2025 bear)
- 1d HTF bias prevents counter-trend trades in strong moves
- Choppiness regime switch adapts to market conditions
- Moderate RSI thresholds (40/60) ensure sufficient trade generation
- 12h TF = 25-40 trades/year target (low fee drag, enough samples)

Timeframe: 12h (REQUIRED)
HTF: 1d via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.30 discrete (max 0.35)
Stoploss: 2.5 * ATR(14) trailing
Target trades: 25-40/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_rsi_chop_regime_1d_v1"
timeframe = "12h"
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

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts to market volatility - moves fast in trends, slow in chop.
    """
    close_s = pd.Series(close)
    n = len(close)
    kama = np.zeros(n)
    
    # Change = abs(close - close[period] ago)
    change = np.abs(close_s - close_s.shift(er_period)).values
    
    # Volatility = sum of abs(close - close[1] ago) over period
    volatility = pd.Series(np.abs(close_s.diff())).rolling(window=er_period, min_periods=er_period).sum().values
    
    # Efficiency Ratio (ER) = Change / Volatility (0 to 1)
    er = change / np.where(volatility > 0, volatility, 1e-10)
    er = np.clip(er, 0, 1)
    
    # Smoothing Constant (SC) = [ER * (fast - slow) + slow]^2
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama[er_period] = close[er_period]
    for i in range(er_period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_hma_slope(hma_values, lookback=5):
    """Calculate HMA slope as percentage change."""
    slope = np.zeros(len(hma_values))
    for i in range(lookback, len(hma_values)):
        if hma_values[i - lookback] != 0:
            slope[i] = (hma_values[i] - hma_values[i - lookback]) / hma_values[i - lookback] * 100
    return slope

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = range market (mean revert)
    CHOP < 38.2 = trend market (trend follow)
    """
    atr_values = calculate_atr(high, low, close, period)
    
    atr_sum = pd.Series(atr_values).rolling(window=period, min_periods=period).sum().values
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    price_range = highest_high - lowest_low
    price_range = np.where(price_range == 0, 1e-10, price_range)
    
    chop = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    chop = np.clip(chop, 0, 100)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1d_slope = calculate_hma_slope(hma_1d_21, 5)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_slope)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    kama_12h = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    chop_14 = calculate_choppiness(high, low, close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.30
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_slope_aligned[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(rsi_14[i]) or np.isnan(kama_12h[i]):
            continue
        
        # === 1D TREND BIAS ===
        # More lenient slope thresholds to allow more trades
        trend_1d_bullish = hma_1d_slope_aligned[i] > 0.15
        trend_1d_bearish = hma_1d_slope_aligned[i] < -0.15
        price_above_1d_hma = close[i] > hma_1d_21_aligned[i]
        price_below_1d_hma = close[i] < hma_1d_21_aligned[i]
        
        # === CHOPPINESS REGIME ===
        is_range_market = chop_14[i] > 55
        is_trend_market = chop_14[i] < 45
        # Neutral zone: 45-55 (use smaller size)
        is_neutral = not is_range_market and not is_trend_market
        
        # === KAMA TREND ===
        price_above_kama = close[i] > kama_12h[i]
        price_below_kama = close[i] < kama_12h[i]
        
        # KAMA slope (simple 5-bar lookback)
        kama_slope = 0.0
        if i >= 5 and kama_12h[i-5] != 0:
            kama_slope = (kama_12h[i] - kama_12h[i-5]) / kama_12h[i-5] * 100
        kama_bullish = kama_slope > 0.1
        kama_bearish = kama_slope < -0.1
        
        # === RSI SIGNALS ===
        rsi_oversold = rsi_14[i] < 40
        rsi_overbought = rsi_14[i] > 60
        rsi_extreme_low = rsi_14[i] < 30
        rsi_extreme_high = rsi_14[i] > 70
        rsi_neutral_low = rsi_14[i] < 45
        rsi_neutral_high = rsi_14[i] > 55
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        if is_neutral:
            current_size = BASE_SIZE * 0.6
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES - Multiple paths for trade generation
        long_signal = False
        
        # Path 1: Trend market + 1d bullish + KAMA bullish + RSI pullback
        if is_trend_market and trend_1d_bullish and kama_bullish and rsi_neutral_low:
            long_signal = True
        
        # Path 2: Range market + RSI oversold (mean revert)
        if is_range_market and rsi_oversold:
            long_signal = True
        
        # Path 3: Price above 1d HMA + RSI pullback (bull trend continuation)
        if price_above_1d_hma and rsi_14[i] < 45 and kama_bullish:
            long_signal = True
        
        # Path 4: KAMA cross above + RSI confirmation
        if price_above_kama and kama_bullish and rsi_14[i] < 50:
            long_signal = True
        
        # Path 5: 1d bullish bias alone (simpler, more trades)
        if trend_1d_bullish and rsi_14[i] < 40:
            long_signal = True
        
        if long_signal:
            new_signal = current_size
        
        # SHORT ENTRIES
        short_signal = False
        
        # Path 1: Trend market + 1d bearish + KAMA bearish + RSI pullback
        if is_trend_market and trend_1d_bearish and kama_bearish and rsi_neutral_high:
            short_signal = True
        
        # Path 2: Range market + RSI overbought (mean revert)
        if is_range_market and rsi_overbought:
            short_signal = True
        
        # Path 3: Price below 1d HMA + RSI rally (bear trend continuation)
        if price_below_1d_hma and rsi_14[i] > 55 and kama_bearish:
            short_signal = True
        
        # Path 4: KAMA cross below + RSI confirmation
        if price_below_kama and kama_bearish and rsi_14[i] > 50:
            short_signal = True
        
        # Path 5: 1d bearish bias alone (simpler, more trades)
        if trend_1d_bearish and rsi_14[i] > 60:
            short_signal = True
        
        if short_signal:
            new_signal = -current_size
        
        # === TRADE FREQUENCY BOOSTER ===
        # If no trades for 100 bars (~50 days), force entry on weaker signals
        if bars_since_last_trade > 100 and new_signal == 0.0 and not in_position:
            if trend_1d_bullish and rsi_14[i] < 50:
                new_signal = current_size * 0.5
            elif trend_1d_bearish and rsi_14[i] > 50:
                new_signal = -current_size * 0.5
            elif rsi_14[i] < 35:
                new_signal = current_size * 0.4
            elif rsi_14[i] > 65:
                new_signal = -current_size * 0.4
        
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
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Exit long if 1d trend turns bearish
            if position_side > 0 and trend_1d_bearish and price_below_1d_hma:
                trend_reversal = True
            # Exit short if 1d trend turns bullish
            if position_side < 0 and trend_1d_bullish and price_above_1d_hma:
                trend_reversal = True
        
        if stoploss_triggered or trend_reversal:
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