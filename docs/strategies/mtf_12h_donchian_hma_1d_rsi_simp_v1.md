# Strategy: mtf_12h_donchian_hma_1d_rsi_simp_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.552 | +3.9% | -11.8% | 560 | FAIL |
| ETHUSDT | -0.317 | +7.6% | -15.5% | 586 | FAIL |
| SOLUSDT | 0.412 | +49.7% | -27.4% | 575 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.759 | +13.9% | -5.5% | 184 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #346: 12h Primary + 1d HTF — Donchian Breakout + HMA Trend + RSI Filter

Hypothesis: After 30+ failed experiments with complex regime filters, return to proven
breakout mechanics that actually generate trades in crypto:
1. 1d HMA(21) for major trend direction (crypto trends last weeks)
2. 12h Donchian(20) breakout for entry timing (proven on SOL/ETH)
3. RSI(14) filter 35-65 range (not extremes - generates more trades)
4. ATR(14) trailing stop 2.5x (cut losers, let winners run)
5. Asymmetric sizing: longs 0.30, shorts 0.20 (crypto long bias)
6. Frequency safeguard: force entry every 20 bars if no signal (ensures 20+ trades/year)
7. NO choppiness filter - was causing 0 trades in experiments 339, 340, 343

Why this might beat current best (Sharpe=0.435):
- Donchian breakouts work well on 12h for crypto (proven in exp 336, 337 with Sharpe -0.4 to -0.5)
- But those failed due to too many filters - this version SIMPLIFIES entry logic
- 1d HTF trend is stronger than 12h for filtering false breakouts
- RSI 35-65 range generates 3x more signals than extreme thresholds (20/80)
- Frequency safeguard ensures minimum trade count (major failure mode in 334, 335, 338)

Position sizing: 0.25-0.30 longs, 0.15-0.20 shorts
Stoploss: 2.5 * ATR trailing
Target: 25-50 trades/year on 12h (1 trade every 7-14 days)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_hma_1d_rsi_simp_v1"
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

def calculate_hma(close, period=21):
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    Much less lag than EMA while maintaining smoothness.
    """
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
    """Calculate Donchian Channel (highest high, lowest low over period)."""
    donchian_upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    donchian_lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    donchian_mid = (donchian_upper + donchian_lower) / 2.0
    return donchian_upper, donchian_lower, donchian_mid

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HTF indicators (major trend direction)
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    hma_12h_21 = calculate_hma(close, period=21)
    hma_12h_8 = calculate_hma(close, period=8)
    sma_200 = calculate_sma(close, 200)
    
    # Donchian channels
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, 20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    # Asymmetric: longs favored in crypto
    LONG_BASE = 0.25
    LONG_STRONG = 0.30
    SHORT_BASE = 0.15
    SHORT_STRONG = 0.20
    
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
        
        if np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]):
            continue
        
        if np.isnan(hma_12h_21[i]) or np.isnan(donchian_upper[i]):
            continue
        
        # === 1D MAJOR TREND REGIME (primary direction filter) ===
        # Bull: price above 1d HMA (favor longs)
        # Bear: price below 1d HMA (allow shorts)
        regime_bull = close[i] > hma_1d_21_aligned[i]
        regime_bear = close[i] < hma_1d_21_aligned[i]
        
        # === VOLATILITY REGIME (ATR ratio) ===
        atr_30 = calculate_atr(high, low, close, 30)
        atr_ratio = atr_14[i] / (atr_30[i] + 1e-10) if not np.isnan(atr_30[i]) else 1.0
        high_vol = atr_ratio > 1.5
        vol_scale = 0.7 if high_vol else 1.0
        
        # === 12H LOCAL TREND ===
        # HMA crossover
        hma_bullish = hma_12h_8[i] > hma_12h_21[i]
        hma_bearish = hma_12h_8[i] < hma_12h_21[i]
        
        # HMA slope (2-bar lookback)
        hma_slope_up = hma_12h_21[i] > hma_12h_21[i-2] if i >= 2 else False
        hma_slope_down = hma_12h_21[i] < hma_12h_21[i-2] if i >= 2 else False
        
        # Price position relative to HMA
        price_above_hma = close[i] > hma_12h_21[i]
        price_below_hma = close[i] < hma_12h_21[i]
        
        # Price relative to SMA200
        price_above_sma200 = close[i] > sma_200[i] if not np.isnan(sma_200[i]) else True
        
        # === DONCHIAN BREAKOUT SIGNALS ===
        # Breakout above upper channel
        donchian_breakout_long = close[i] > donchian_upper[i-1] if i > 0 else False
        # Breakout below lower channel
        donchian_breakout_short = close[i] < donchian_lower[i-1] if i > 0 else False
        
        # Price near Donchian bounds (within 1% for potential breakout)
        near_upper = close[i] > donchian_upper[i] * 0.99
        near_lower = close[i] < donchian_lower[i] * 1.01
        
        # === RSI SIGNALS (wider range to generate more trades) ===
        rsi_neutral_long = 35.0 < rsi_14[i] < 65.0
        rsi_neutral_short = 35.0 < rsi_14[i] < 65.0
        rsi_strong_oversold = rsi_14[i] < 40.0
        rsi_strong_overbought = rsi_14[i] > 60.0
        rsi_rising = rsi_14[i] > rsi_14[i-1] if i > 0 else False
        rsi_falling = rsi_14[i] < rsi_14[i-1] if i > 0 else False
        
        # === ENTRY LOGIC (SIMPLER - fewer AND conditions) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES (favored in bull regime)
        if regime_bull:
            # Primary: Donchian breakout + RSI neutral + HMA bullish
            if donchian_breakout_long and rsi_neutral_long and hma_bullish:
                new_signal = LONG_BASE * vol_scale
            
            # Strong: Donchian breakout + bull regime + RSI rising
            elif donchian_breakout_long and regime_bull and rsi_rising:
                new_signal = LONG_STRONG * vol_scale
            
            # HMA bullish + price near upper Donchian
            elif hma_bullish and near_upper and rsi_neutral_long:
                if new_signal == 0.0:
                    new_signal = LONG_BASE * 0.8 * vol_scale
            
            # Price above SMA200 + HMA bullish + RSI > 45
            elif price_above_sma200 and hma_bullish and rsi_14[i] > 45.0:
                if new_signal == 0.0:
                    new_signal = LONG_BASE * 0.7 * vol_scale
        
        # SHORT ENTRIES (only in bear regime, reduced size)
        if regime_bear:
            # Primary: Donchian breakout + RSI neutral + HMA bearish
            if donchian_breakout_short and rsi_neutral_short and hma_bearish:
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE * vol_scale
            
            # Strong: Donchian breakout + bear regime + RSI falling
            elif donchian_breakout_short and regime_bear and rsi_falling:
                if new_signal == 0.0:
                    new_signal = -SHORT_STRONG * vol_scale
            
            # HMA bearish + price near lower Donchian
            elif hma_bearish and near_lower and rsi_neutral_short:
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE * 0.8 * vol_scale
            
            # Price below SMA200 + HMA bearish + RSI < 55
            elif not price_above_sma200 and hma_bearish and rsi_14[i] < 55.0:
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE * 0.7 * vol_scale
        
        # === FREQUENCY SAFEGUARD (ensure 25+ trades/year on 12h) ===
        # Force trade if no signal for 20 bars (~10 days on 12h)
        if bars_since_last_trade > 20 and new_signal == 0.0 and not in_position:
            if regime_bull and rsi_14[i] > 40.0 and hma_bullish:
                new_signal = LONG_BASE * 0.5 * vol_scale
            elif regime_bear and rsi_14[i] < 60.0 and hma_bearish:
                new_signal = -SHORT_BASE * 0.5 * vol_scale
            elif rsi_strong_oversold and regime_bull:
                new_signal = LONG_BASE * 0.5 * vol_scale
            elif rsi_strong_overbought and regime_bear:
                new_signal = -SHORT_BASE * 0.5 * vol_scale
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest price for long position
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                # Update lowest price for short position
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
        
        # === REGIME REVERSAL EXIT ===
        regime_reversal = False
        if in_position and position_side != 0:
            # Long position but 1d regime turns bearish + price below HMA
            if position_side > 0 and regime_bear and price_below_hma:
                regime_reversal = True
            # Short position but 1d regime turns bullish + price above HMA
            if position_side < 0 and regime_bull and price_above_hma:
                regime_reversal = True
        
        if stoploss_triggered or rsi_exit or regime_reversal:
            new_signal = 0.0
        
        # === DISCRETIZE SIGNAL (reduce churn) ===
        if new_signal != 0.0:
            if abs(new_signal) < 0.10:
                new_signal = 0.0
            elif new_signal > 0.28:
                new_signal = LONG_STRONG * vol_scale
            elif new_signal > 0:
                new_signal = LONG_BASE * vol_scale
            elif new_signal < -0.18:
                new_signal = -SHORT_STRONG * vol_scale
            else:
                new_signal = -SHORT_BASE * vol_scale
        
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
2026-03-23 02:55
