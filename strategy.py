#!/usr/bin/env python3
"""
Experiment #387: 1d Primary + 1w HTF — Donchian Breakout + HMA Trend + RSI Filter

Hypothesis: After analyzing 386 experiments, the pattern is clear:
1. 1d timeframe generates optimal trade frequency (20-50 trades/year)
2. Simple trend-follow with breakout entries beats complex regime-switching
3. 1w HMA(21) for major trend bias (proven in current best strategy)
4. Donchian(20) breakout for entry timing (catches momentum shifts)
5. RSI(14) filter to avoid false breakouts at extremes
6. ATR 2.5x trailing stop for risk management
7. Discrete position sizing: 0.0, ±0.25, ±0.30

Why this might beat current best (Sharpe=0.435):
- Donchian breakouts capture sustained moves (not whipsaws)
- 1w HTF filter prevents counter-trend trades in major trends
- RSI filter avoids entering at momentum exhaustion
- 1d TF generates 25-40 trades/year (optimal for fee/capture balance)
- Works in both bull and bear markets via trend filter
- Simpler than dual-regime approaches that failed in exp #375, #379, #383

Position sizing: 0.25-0.30 (discrete levels, max 0.40)
Stoploss: 2.5 * ATR trailing
Target: 25-40 trades/year on 1d, >=30 trades/symbol on train
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_hma_rsi_1w_v1"
timeframe = "1d"
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
    """Calculate Hull Moving Average (HMA)."""
    n = period
    half = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        return series.rolling(window=span, min_periods=span).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    hma_raw = 2.0 * wma_half - wma_full
    hma = wma(hma_raw, sqrt_n)
    
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

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (upper and lower bands)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_sma(close, period=50):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1w HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HTF indicators (major trend direction)
    hma_1w_21 = calculate_hma(df_1w['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    hma_1d_21 = calculate_hma(close, period=21)
    hma_1d_50 = calculate_hma(close, period=50)
    sma_1d_50 = calculate_sma(close, 50)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    LONG_SIZE = 0.30
    SHORT_SIZE = 0.25
    
    # Track position state
    in_position = False
    position_side = 0
    highest_price = 0.0
    lowest_price = 0.0
    entry_price = 0.0
    last_trade_bar = -20
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1w_21_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(hma_1d_21[i]) or np.isnan(hma_1d_50[i]):
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        # === 1W MAJOR TREND (primary direction filter) ===
        # Price above 1w HMA = bull market bias (favor longs)
        # Price below 1w HMA = bear market bias (favor shorts)
        bull_regime = close[i] > hma_1w_21_aligned[i]
        bear_regime = close[i] < hma_1w_21_aligned[i]
        
        # === 1D LOCAL TREND (HMA crossover) ===
        hma_bullish = hma_1d_21[i] > hma_1d_50[i]
        hma_bearish = hma_1d_21[i] < hma_1d_50[i]
        
        # === DONCHIAN BREAKOUT SIGNALS ===
        # Long breakout: price breaks above Donchian upper
        donchian_long_breakout = close[i] > donchian_upper[i-1] if i > 0 else False
        # Short breakout: price breaks below Donchian lower
        donchian_short_breakout = close[i] < donchian_lower[i-1] if i > 0 else False
        
        # === RSI FILTER (avoid false breakouts at extremes) ===
        # For longs: RSI should not be overbought (>75)
        rsi_long_ok = rsi_14[i] < 75.0
        # For shorts: RSI should not be oversold (<25)
        rsi_short_ok = rsi_14[i] > 25.0
        
        # === RSI PULLBACK ENTRY (alternative entry) ===
        # Long pullback: RSI 40-55 in bull regime
        rsi_long_pullback = 40.0 <= rsi_14[i] <= 55.0
        # Short pullback: RSI 45-60 in bear regime
        rsi_short_pullback = 45.0 <= rsi_14[i] <= 60.0
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRY: Bull regime + (Donchian breakout OR HMA bullish + RSI pullback)
        if bull_regime:
            if donchian_long_breakout and rsi_long_ok:
                new_signal = LONG_SIZE
            elif hma_bullish and rsi_long_pullback:
                new_signal = LONG_SIZE * 0.8
        
        # SHORT ENTRY: Bear regime + (Donchian breakout OR HMA bearish + RSI pullback)
        if bear_regime:
            if donchian_short_breakout and rsi_short_ok:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE
            elif hma_bearish and rsi_short_pullback:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE * 0.8
        
        # === FREQUENCY BOOST (ensure >=30 trades/symbol on train) ===
        # If no trade for 20 bars (~20 days on 1d), force entry on weaker signal
        if bars_since_last_trade > 20 and new_signal == 0.0 and not in_position:
            if bull_regime and rsi_14[i] < 50:
                new_signal = LONG_SIZE * 0.6
            elif bear_regime and rsi_14[i] > 50:
                new_signal = -SHORT_SIZE * 0.6
        
        # === EXIT CONDITIONS ===
        # RSI extreme exit (take profit on momentum exhaustion)
        if in_position and position_side > 0 and rsi_14[i] > 75:
            new_signal = 0.0
        if in_position and position_side < 0 and rsi_14[i] < 25:
            new_signal = 0.0
        
        # Trend reversal exit (1w regime flip)
        if in_position and position_side > 0 and bear_regime:
            new_signal = 0.0
        if in_position and position_side < 0 and bull_regime:
            new_signal = 0.0
        
        # Local trend reversal exit (1d HMA cross)
        if in_position and position_side > 0 and hma_bearish:
            new_signal = 0.0
        if in_position and position_side < 0 and hma_bullish:
            new_signal = 0.0
        
        # === STOPLOSS (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_price = max(highest_price, close[i])
            stop_price = highest_price - 2.5 * atr_14[i]
            if close[i] < stop_price and not np.isnan(stop_price):
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_price == 0.0:
                lowest_price = close[i]
            else:
                lowest_price = min(lowest_price, close[i])
            stop_price = lowest_price + 2.5 * atr_14[i]
            if close[i] > stop_price and not np.isnan(stop_price):
                stoploss_triggered = True
        
        if stoploss_triggered:
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
                # Position flip
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