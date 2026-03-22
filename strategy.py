#!/usr/bin/env python3
"""
Experiment #141: 4h Primary + 1d/1w HTF — KAMA Adaptive Trend + RSI Entries

Hypothesis: Previous 4h strategies failed due to over-filtering (0 trades) or 
pure trend-following (negative in bear markets). This strategy uses:

1. KAMA (Kaufman Adaptive MA): Adapts to market efficiency - fast in trends, 
   slow in chop. Better than EMA/HMA for crypto's regime changes.
2. ADX Regime Filter: ADX>25 = trend, ADX<20 = range. Different logic per regime.
3. RSI Entries: Simpler than Connors, more reliable. Long RSI<40, Short RSI>60.
4. 1d HTF Bias: 1d KAMA slope determines preferred direction (not hard filter).
5. ATR Stops: 2.5*ATR trailing stop on all positions.

Why this should work:
- KAMA adapts to volatility changes better than fixed-period MAs
- ADX regime detection allows different strategies for different markets
- RSI thresholds are less extreme = more trades (target 30-50/year)
- 1d bias is soft, not hard - allows counter-trend in strong mean-reversion
- 4h timeframe balances trade frequency with signal quality

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.30 discrete
Stoploss: 2.5 * ATR(14) trailing
Target trades: 30-50/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_adx_rsi_regime_1d_v1"
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

def calculate_kama(close, er_period=10, fast_sc=2, slow_sc=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts to market noise - fast during trends, slow during chop.
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Change = |Close - Close 10 periods ago|
    change = np.abs(close - np.roll(close, er_period))
    change[:er_period] = np.nan
    
    # Volatility = Sum of |Close - Prior Close| for 10 periods
    diff = np.abs(close_s.diff())
    volatility = pd.Series(diff).rolling(window=er_period, min_periods=er_period).sum().values
    
    # Efficiency Ratio (ER) = Change / Volatility
    er = change / np.where(volatility > 0, volatility, 1e-10)
    er = np.nan_to_num(er, nan=0.0)
    er = np.clip(er, 0, 1)
    
    # Smoothing Constant (SC) = [ER * (fast_sc - slow_sc) + slow_sc]^2
    fast_const = 2.0 / (fast_sc + 1)
    slow_const = 2.0 / (slow_sc + 1)
    sc = np.power(er * (fast_const - slow_const) + slow_const, 2)
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[0] = close[0]
    
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean() / atr
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.fillna(0).values

def calculate_kama_slope(kama_values, lookback=5):
    """Calculate KAMA slope as percentage change."""
    slope = np.zeros(len(kama_values))
    for i in range(lookback, len(kama_values)):
        if kama_values[i - lookback] != 0:
            slope[i] = (kama_values[i] - kama_values[i - lookback]) / kama_values[i - lookback] * 100
    return slope

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    kama_1d = calculate_kama(df_1d['close'].values, 10)
    kama_1d_slope = calculate_kama_slope(kama_1d, 5)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    kama_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_slope)
    
    # Calculate 4h indicators
    kama_4h = calculate_kama(close, 10)
    kama_4h_fast = calculate_kama(close, 5)  # Faster KAMA for crossovers
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    adx_14 = calculate_adx(high, low, close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.20
    
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
        
        if np.isnan(kama_1d_aligned[i]) or np.isnan(kama_1d_slope_aligned[i]):
            continue
        
        if np.isnan(kama_4h[i]) or np.isnan(rsi_14[i]) or np.isnan(adx_14[i]):
            continue
        
        # === 1D TREND BIAS (soft filter, not hard) ===
        trend_1d_bullish = kama_1d_slope_aligned[i] > 0.2
        trend_1d_bearish = kama_1d_slope_aligned[i] < -0.2
        price_above_1d_kama = close[i] > kama_1d_aligned[i]
        price_below_1d_kama = close[i] < kama_1d_aligned[i]
        
        # === ADX REGIME DETECTION ===
        is_trending = adx_14[i] > 25
        is_ranging = adx_14[i] < 20
        
        # === KAMA TREND (4h) ===
        kama_bullish = kama_4h_fast[i] > kama_4h[i]
        kama_bearish = kama_4h_fast[i] < kama_4h[i]
        price_above_kama = close[i] > kama_4h[i]
        price_below_kama = close[i] < kama_4h[i]
        
        # === RSI LEVELS ===
        rsi_oversold = rsi_14[i] < 45
        rsi_overbought = rsi_14[i] > 55
        rsi_extreme_low = rsi_14[i] < 35
        rsi_extreme_high = rsi_14[i] > 65
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        if is_ranging:
            current_size = REDUCED_SIZE  # Smaller size in chop
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES - Multiple paths for sufficient trade frequency
        long_score = 0
        
        # Path 1: Trending + KAMA bullish + RSI pullback (trend follow)
        if is_trending and kama_bullish and rsi_oversold:
            long_score += 3
        
        # Path 2: Ranging + RSI extreme low (mean revert)
        if is_ranging and rsi_extreme_low:
            long_score += 2
        
        # Path 3: 1d bullish bias + 4h RSI oversold (HTF alignment)
        if trend_1d_bullish and rsi_14[i] < 40:
            long_score += 2
        
        # Path 4: Price above 4h KAMA + RSI dip (pullback in uptrend)
        if price_above_kama and rsi_14[i] < 45 and kama_bullish:
            long_score += 2
        
        # Path 5: Simple RSI extreme (fallback for more trades)
        if rsi_extreme_low and price_above_1d_kama:
            long_score += 1
        
        # Path 6: KAMA crossover + RSI confirmation
        if kama_bullish and rsi_14[i] < 50 and adx_14[i] > 15:
            long_score += 1
        
        if long_score >= 3:
            new_signal = current_size
        elif long_score == 2 and bars_since_last_trade > 40:
            new_signal = current_size
        elif long_score >= 1 and bars_since_last_trade > 80:
            new_signal = REDUCED_SIZE
        
        # SHORT ENTRIES
        short_score = 0
        
        # Path 1: Trending + KAMA bearish + RSI rally (trend follow)
        if is_trending and kama_bearish and rsi_overbought:
            short_score += 3
        
        # Path 2: Ranging + RSI extreme high (mean revert)
        if is_ranging and rsi_extreme_high:
            short_score += 2
        
        # Path 3: 1d bearish bias + 4h RSI overbought (HTF alignment)
        if trend_1d_bearish and rsi_14[i] > 60:
            short_score += 2
        
        # Path 4: Price below 4h KAMA + RSI spike (pullback in downtrend)
        if price_below_kama and rsi_14[i] > 55 and kama_bearish:
            short_score += 2
        
        # Path 5: Simple RSI extreme (fallback for more trades)
        if rsi_extreme_high and price_below_1d_kama:
            short_score += 1
        
        # Path 6: KAMA crossover + RSI confirmation
        if kama_bearish and rsi_14[i] > 50 and adx_14[i] > 15:
            short_score += 1
        
        if short_score >= 3:
            new_signal = -current_size
        elif short_score == 2 and bars_since_last_trade > 40:
            new_signal = -current_size
        elif short_score >= 1 and bars_since_last_trade > 80:
            new_signal = -REDUCED_SIZE
        
        # === FREQUENCY SAFEGUARD ===
        # Force trade if no signal for 120 bars (~20 days on 4h)
        if bars_since_last_trade > 120 and new_signal == 0.0 and not in_position:
            if trend_1d_bullish and rsi_14[i] < 45:
                new_signal = REDUCED_SIZE
            elif trend_1d_bearish and rsi_14[i] > 55:
                new_signal = -REDUCED_SIZE
            elif rsi_extreme_low:
                new_signal = REDUCED_SIZE * 0.7
            elif rsi_extreme_high:
                new_signal = -REDUCED_SIZE * 0.7
        
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
            # Exit long if regime shifts to strong bearish trend
            if position_side > 0 and is_trending and trend_1d_bearish and kama_bearish:
                regime_reversal = True
            # Exit short if regime shifts to strong bullish trend
            if position_side < 0 and is_trending and trend_1d_bullish and kama_bullish:
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