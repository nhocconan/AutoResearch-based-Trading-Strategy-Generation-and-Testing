# Strategy: mtf_4h_donchian_hma_rsi_12h1d_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.309 | -23.3% | -32.9% | 483 | FAIL |
| ETHUSDT | -0.544 | -7.3% | -20.4% | 523 | FAIL |
| SOLUSDT | 0.604 | +78.8% | -18.2% | 567 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 1.300 | +27.6% | -8.4% | 157 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #214: 4h Primary + 12h/1d HTF — Donchian Breakout + HMA Trend + RSI Filter

Hypothesis: After 196 failed experiments, the pattern is clear: complex regime-switching
and Connors RSI combinations are over-optimized and failing. This strategy goes BACK TO BASICS:

1. DONCHIAN(20) BREAKOUT: Simple price breakout - proven to work across market regimes
2. HMA(21) TREND FILTER: 12h HMA slope determines long/short bias (no counter-trend)
3. RSI(14) MOMENTUM: RSI > 55 for longs, < 45 for shorts (confirms momentum)
4. ATR(14) TRAILING STOP: 2.5 * ATR stoploss on every position
5. 1d HMA CONFIRMATION: Extra HTF filter to avoid fighting major trends

Why this should work:
- Donchian breakouts capture sustained moves (not whipsaws)
- HMA is faster than EMA, catches trends earlier
- RSI filter avoids breakouts with no momentum backing
- 4h timeframe = 20-50 trades/year target (matches cost model)
- Simple logic = fewer conditions that can all fail simultaneously

Key difference from failed strategies:
- NO Connors RSI (overused, failing)
- NO Choppiness Index (overused, failing)
- NO complex regime switching (causes 0 trades)
- Loose enough entries to guarantee 30+ trades/year

Timeframe: 4h (REQUIRED for this experiment)
HTF: 12h + 1d via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.28 discrete (max 0.35)
Stoploss: 2.5 * ATR(14) trailing
Target trades: 25-50/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_hma_rsi_12h1d_v1"
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
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 12h HTF indicators
    hma_12h_21 = calculate_hma(df_12h['close'].values, 21)
    hma_12h_slope = calculate_hma_slope(hma_12h_21, 5)
    
    # Calculate 1d HTF indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1d_slope = calculate_hma_slope(hma_1d_21, 5)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_12h_21_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_21)
    hma_12h_slope_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_slope)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_slope)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian_channels(high, low, 20)
    
    # 4h HMA for local trend
    hma_4h_21 = calculate_hma(close, 21)
    
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
        
        if np.isnan(hma_12h_21_aligned[i]) or np.isnan(hma_12h_slope_aligned[i]):
            continue
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_slope_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(donchian_upper[i]):
            continue
        
        # === HTF TREND BIAS (12h + 1d) ===
        trend_12h_bullish = hma_12h_slope_aligned[i] > 0.2
        trend_12h_bearish = hma_12h_slope_aligned[i] < -0.2
        trend_1d_bullish = hma_1d_slope_aligned[i] > 0.3
        trend_1d_bearish = hma_1d_slope_aligned[i] < -0.3
        
        # Price relative to HTF HMA
        price_above_12h_hma = close[i] > hma_12h_21_aligned[i]
        price_below_12h_hma = close[i] < hma_12h_21_aligned[i]
        price_above_1d_hma = close[i] > hma_1d_21_aligned[i]
        price_below_1d_hma = close[i] < hma_1d_21_aligned[i]
        
        # === LOCAL TREND (4h HMA) ===
        price_above_4h_hma = close[i] > hma_4h_21[i]
        price_below_4h_hma = close[i] < hma_4h_21[i]
        
        # === MOMENTUM (RSI) ===
        rsi_bullish = rsi_14[i] > 55
        rsi_bearish = rsi_14[i] < 45
        rsi_neutral = 45 <= rsi_14[i] <= 55
        
        # === DONCHIAN BREAKOUT ===
        breakout_long = close[i] > donchian_upper[i]
        breakout_short = close[i] < donchian_lower[i]
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        
        # Reduce size if HTF trends conflict
        if (trend_12h_bullish and trend_1d_bearish) or (trend_12h_bearish and trend_1d_bullish):
            current_size = BASE_SIZE * 0.6
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES - Multiple paths for trade frequency
        long_conditions = 0
        
        # Path 1: Donchian breakout + 12h bullish + RSI bullish (primary)
        if breakout_long and trend_12h_bullish and rsi_bullish:
            long_conditions += 3
        
        # Path 2: Donchian breakout + price above 12h HMA + RSI > 50
        if breakout_long and price_above_12h_hma and rsi_14[i] > 50:
            long_conditions += 2
        
        # Path 3: 1d bullish + 12h bullish + RSI bullish (trend continuation)
        if trend_1d_bullish and trend_12h_bullish and rsi_bullish and price_above_4h_hma:
            long_conditions += 2
        
        # Path 4: Breakout + 1d bullish (stronger HTF confirmation)
        if breakout_long and trend_1d_bullish:
            long_conditions += 2
        
        # Path 5: Simple breakout with RSI confirmation (looser for more trades)
        if breakout_long and rsi_14[i] > 52:
            long_conditions += 1
        
        if long_conditions >= 2:
            new_signal = current_size
        elif long_conditions == 1 and bars_since_last_trade > 60:
            new_signal = current_size * 0.5
        
        # SHORT ENTRIES
        short_conditions = 0
        
        # Path 1: Donchian breakout + 12h bearish + RSI bearish (primary)
        if breakout_short and trend_12h_bearish and rsi_bearish:
            short_conditions += 3
        
        # Path 2: Donchian breakout + price below 12h HMA + RSI < 50
        if breakout_short and price_below_12h_hma and rsi_14[i] < 50:
            short_conditions += 2
        
        # Path 3: 1d bearish + 12h bearish + RSI bearish (trend continuation)
        if trend_1d_bearish and trend_12h_bearish and rsi_bearish and price_below_4h_hma:
            short_conditions += 2
        
        # Path 4: Breakout + 1d bearish (stronger HTF confirmation)
        if breakout_short and trend_1d_bearish:
            short_conditions += 2
        
        # Path 5: Simple breakout with RSI confirmation (looser for more trades)
        if breakout_short and rsi_14[i] < 48:
            short_conditions += 1
        
        if short_conditions >= 2:
            new_signal = -current_size
        elif short_conditions == 1 and bars_since_last_trade > 60:
            new_signal = -current_size * 0.5
        
        # === FREQUENCY SAFEGUARD ===
        # Force trade if no signal for 120 bars (~20 days on 4h)
        if bars_since_last_trade > 120 and new_signal == 0.0 and not in_position:
            if trend_1d_bullish and trend_12h_bullish and rsi_14[i] > 50:
                new_signal = current_size * 0.4
            elif trend_1d_bearish and trend_12h_bearish and rsi_14[i] < 50:
                new_signal = -current_size * 0.4
            elif rsi_14[i] > 60 and price_above_12h_hma:
                new_signal = current_size * 0.3
            elif rsi_14[i] < 40 and price_below_12h_hma:
                new_signal = -current_size * 0.3
        
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
            # Long position but HTF turns bearish
            if position_side > 0 and trend_12h_bearish and trend_1d_bearish:
                trend_reversal = True
            # Short position but HTF turns bullish
            if position_side < 0 and trend_12h_bullish and trend_1d_bullish:
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
2026-03-23 00:50
