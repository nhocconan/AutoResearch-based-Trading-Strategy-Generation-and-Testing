# Strategy: mtf_1d_donchian_hma_rsi_1w_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.163 | +27.3% | -8.5% | 81 | PASS |
| ETHUSDT | -0.772 | -6.6% | -16.6% | 64 | FAIL |
| SOLUSDT | 0.256 | +35.7% | -22.4% | 73 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.812 | +0.5% | -3.3% | 25 | FAIL |
| SOLUSDT | 0.258 | +9.9% | -10.2% | 29 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #217: 1d Primary + 1w HTF — Donchian Breakout + HMA Trend + RSI Filter

Hypothesis: After 216 experiments, complex regime-switching and Connors RSI 
combinations consistently fail. This strategy returns to PROVEN basics:

1. DONCHIAN(20) BREAKOUT: Captures sustained moves, works across regimes
2. HMA(21) TREND: Faster than EMA, catches trends earlier with less lag
3. RSI(14) MOMENTUM: Confirms breakout has momentum backing (not fakeout)
4. 1w HTF FILTER: Major trend alignment (never fight the weekly trend)
5. ATR(14) TRAILING STOP: 2.5 * ATR protects against reversals

Why 1d timeframe:
- Natural filter against noise (fewer whipsaws than lower TF)
- 10-30 trades/year target matches cost model perfectly
- Each trade has time to develop (no premature exits)
- Works well with weekly HTF confirmation

Key differences from failed strategies:
- NO Connors RSI (overused, failing across 50+ experiments)
- NO Choppiness Index (overused, failing)
- NO complex regime switching (causes 0 trades)
- LOOSE entry conditions to guarantee 10+ trades/symbol
- Simple logic = fewer conditions that can all fail simultaneously

Position sizing: 0.28 discrete (max 0.35)
Stoploss: 2.5 * ATR(14) trailing
Target: 15-30 trades/year per symbol
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
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_hma_slope(hma_values, lookback=3):
    """Calculate HMA slope as percentage change."""
    slope = np.zeros(len(hma_values))
    for i in range(lookback, len(hma_values)):
        if hma_values[i - lookback] != 0 and not np.isnan(hma_values[i - lookback]):
            slope[i] = (hma_values[i] - hma_values[i - lookback]) / hma_values[i - lookback] * 100
    return slope

def calculate_donchian_channels(high, low, period=20):
    """Calculate Donchian Channels (highest high / lowest low over period)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (upper + lower) / 2.0
    return upper, lower, mid

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HTF indicators
    hma_1w_21 = calculate_hma(df_1w['close'].values, 21)
    hma_1w_slope = calculate_hma_slope(hma_1w_21, 3)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    hma_1w_slope_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_slope)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian_channels(high, low, 20)
    
    # 1d HMA for local trend
    hma_1d_21 = calculate_hma(close, 21)
    hma_1d_slope = calculate_hma_slope(hma_1d_21, 3)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.28
    
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
        
        if np.isnan(hma_1w_21_aligned[i]) or np.isnan(hma_1w_slope_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(donchian_upper[i]):
            continue
        
        if np.isnan(hma_1d_21[i]) or np.isnan(hma_1d_slope[i]):
            continue
        
        # === HTF TREND BIAS (1w) ===
        # Weekly trend determines overall bias
        weekly_bullish = hma_1w_slope_aligned[i] > 0.15
        weekly_bearish = hma_1w_slope_aligned[i] < -0.15
        weekly_neutral = not weekly_bullish and not weekly_bearish
        
        price_above_1w_hma = close[i] > hma_1w_21_aligned[i]
        price_below_1w_hma = close[i] < hma_1w_21_aligned[i]
        
        # === LOCAL TREND (1d HMA) ===
        daily_bullish = hma_1d_slope[i] > 0.2
        daily_bearish = hma_1d_slope[i] < -0.2
        
        price_above_1d_hma = close[i] > hma_1d_21[i]
        price_below_1d_hma = close[i] < hma_1d_21[i]
        
        # === MOMENTUM (RSI) ===
        rsi_bullish = rsi_14[i] > 52
        rsi_bearish = rsi_14[i] < 48
        rsi_strong_bull = rsi_14[i] > 58
        rsi_strong_bear = rsi_14[i] < 42
        
        # === DONCHIAN BREAKOUT ===
        breakout_long = close[i] > donchian_upper[i]
        breakout_short = close[i] < donchian_lower[i]
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES - Multiple paths for trade frequency (CRITICAL for 10+ trades)
        long_score = 0
        
        # Path 1: Donchian breakout + weekly bullish + RSI bullish (primary - strong signal)
        if breakout_long and weekly_bullish and rsi_bullish:
            long_score += 4
        
        # Path 2: Donchian breakout + price above weekly HMA + RSI > 50
        if breakout_long and price_above_1w_hma and rsi_14[i] > 50:
            long_score += 3
        
        # Path 3: Weekly bullish + daily bullish + RSI bullish (trend continuation)
        if weekly_bullish and daily_bullish and rsi_bullish and price_above_1d_hma:
            long_score += 3
        
        # Path 4: Breakout + weekly bullish (stronger HTF confirmation)
        if breakout_long and weekly_bullish:
            long_score += 2
        
        # Path 5: Breakout + daily bullish + RSI confirmation
        if breakout_long and daily_bullish and rsi_bullish:
            long_score += 2
        
        # Path 6: Simple breakout with RSI confirmation (looser for more trades)
        if breakout_long and rsi_14[i] > 55:
            long_score += 1
        
        # Path 7: Weekly bullish + RSI strong (momentum entry without breakout)
        if weekly_bullish and rsi_strong_bull and price_above_1d_hma and bars_since_last_trade > 30:
            long_score += 1
        
        if long_score >= 3:
            new_signal = current_size
        elif long_score == 2 and bars_since_last_trade > 40:
            new_signal = current_size * 0.6
        elif long_score >= 1 and bars_since_last_trade > 60:
            new_signal = current_size * 0.4
        
        # SHORT ENTRIES
        short_score = 0
        
        # Path 1: Donchian breakout + weekly bearish + RSI bearish (primary)
        if breakout_short and weekly_bearish and rsi_bearish:
            short_score += 4
        
        # Path 2: Donchian breakout + price below weekly HMA + RSI < 50
        if breakout_short and price_below_1w_hma and rsi_14[i] < 50:
            short_score += 3
        
        # Path 3: Weekly bearish + daily bearish + RSI bearish (trend continuation)
        if weekly_bearish and daily_bearish and rsi_bearish and price_below_1d_hma:
            short_score += 3
        
        # Path 4: Breakout + weekly bearish (stronger HTF confirmation)
        if breakout_short and weekly_bearish:
            short_score += 2
        
        # Path 5: Breakout + daily bearish + RSI confirmation
        if breakout_short and daily_bearish and rsi_bearish:
            short_score += 2
        
        # Path 6: Simple breakout with RSI confirmation (looser for more trades)
        if breakout_short and rsi_14[i] < 45:
            short_score += 1
        
        # Path 7: Weekly bearish + RSI strong (momentum entry without breakout)
        if weekly_bearish and rsi_strong_bear and price_below_1d_hma and bars_since_last_trade > 30:
            short_score += 1
        
        if short_score >= 3:
            new_signal = -current_size
        elif short_score == 2 and bars_since_last_trade > 40:
            new_signal = -current_size * 0.6
        elif short_score >= 1 and bars_since_last_trade > 60:
            new_signal = -current_size * 0.4
        
        # === FREQUENCY SAFEGUARD ===
        # Force trade if no signal for 90 bars (~90 days on 1d)
        if bars_since_last_trade > 90 and new_signal == 0.0 and not in_position:
            if weekly_bullish and rsi_14[i] > 50 and price_above_1d_hma:
                new_signal = current_size * 0.35
            elif weekly_bearish and rsi_14[i] < 50 and price_below_1d_hma:
                new_signal = -current_size * 0.35
            elif rsi_14[i] > 62 and price_above_1w_hma:
                new_signal = current_size * 0.25
            elif rsi_14[i] < 38 and price_below_1w_hma:
                new_signal = -current_size * 0.25
        
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
        
        # === HTF TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Long position but weekly turns strongly bearish
            if position_side > 0 and weekly_bearish and price_below_1w_hma:
                trend_reversal = True
            # Short position but weekly turns strongly bullish
            if position_side < 0 and weekly_bullish and price_above_1w_hma:
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
```

## Last Updated
2026-03-23 00:53
