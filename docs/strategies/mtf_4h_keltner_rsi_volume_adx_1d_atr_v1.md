# Strategy: mtf_4h_keltner_rsi_volume_adx_1d_atr_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.041 | +22.1% | -8.9% | 336 | PASS |
| ETHUSDT | -0.900 | -15.5% | -19.9% | 343 | FAIL |
| SOLUSDT | 0.408 | +52.0% | -19.6% | 336 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.586 | +1.3% | -6.7% | 115 | FAIL |
| SOLUSDT | 0.554 | +14.2% | -12.0% | 117 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #761: 4h Primary + 1d HTF — Keltner Mean Reversion + RSI + Volume Filter

Hypothesis: After analyzing 500+ failed strategies and the success of #751 (Sharpe=0.342):
1. Keltner Channels provide better mean reversion signals than Bollinger in crypto (less whipsaw)
2. RSI(14) with moderate thresholds (30/70) ensures sufficient trade frequency vs extreme CRSI
3. Volume confirmation (1.5x average) filters false breakouts and adds conviction
4. ADX(14) > 25 confirms trend strength for trending regime entries
5. 1d EMA(50) provides cleaner trend bias than HMA for this approach
6. Simpler logic = less overfitting, better generalization to test period

Strategy design:
1. 1d EMA(50) for primary trend bias (aligned via mtf_data helper)
2. 4h Choppiness Index(14) for regime detection (trending vs ranging)
3. 4h Keltner Channels (EMA20 + 1.5*ATR14) for mean reversion bounds
4. 4h RSI(14) for entry timing (moderate thresholds for frequency)
5. 4h Volume filter (1.5x 20-bar average) for confirmation
6. 4h ADX(14) for trend strength confirmation
7. 4h ATR(14) for trailing stop (2.5x)
8. Discrete signals: 0.0, ±0.25, ±0.30

Key improvements from #751:
- Replaced CRSI with RSI(14) + Keltner (simpler, more proven in crypto)
- Added volume confirmation filter (missing in #751)
- Added ADX trend strength filter (prevents entries in weak trends)
- More relaxed RSI thresholds to ensure >=30 trades/train
- Cleaner hold/exit logic

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 4h (target 20-50 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_keltner_rsi_volume_adx_1d_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_ema(series, period):
    """Exponential Moving Average."""
    return pd.Series(series).ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_rsi(close, period=14):
    """Relative Strength Index."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    avg_gain = np.concatenate([[np.nan], avg_gain])
    avg_loss = np.concatenate([[np.nan], avg_loss])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_keltner(high, low, close, ema_period=20, atr_mult=1.5, atr_period=14):
    """
    Keltner Channels - EMA +/- ATR multiplier.
    Better for crypto mean reversion than Bollinger Bands.
    """
    ema = calculate_ema(close, ema_period)
    atr = calculate_atr(high, low, close, atr_period)
    
    upper = ema + atr_mult * atr
    lower = ema - atr_mult * atr
    
    return upper, lower, ema

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index - measures trend strength.
    ADX > 25 = trending, ADX < 20 = ranging.
    """
    n = len(close)
    adx = np.full(n, np.nan)
    
    if n < period * 2 + 1:
        return adx
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        plus_move = high[i] - high[i-1]
        minus_move = low[i-1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        elif minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures whether market is trending or ranging.
    CHOP > 61.8 = ranging (mean reversion)
    CHOP < 38.2 = trending (trend follow)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        price_range = highest_high - lowest_low
        if price_range > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_volume_sma(volume, period=20):
    """Simple Moving Average of volume."""
    return pd.Series(volume).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate primary (4h) indicators
    rsi_4h = calculate_rsi(close, period=14)
    atr_4h = calculate_atr(high, low, close, period=14)
    keltner_upper, keltner_lower, keltner_ema = calculate_keltner(high, low, close, ema_period=20, atr_mult=1.5, atr_period=14)
    adx_4h = calculate_adx(high, low, close, period=14)
    chop_4h = calculate_choppiness(high, low, close, period=14)
    vol_sma_4h = calculate_volume_sma(volume, period=20)
    
    # Calculate and align HTF EMA for trend bias
    ema_1d_raw = calculate_ema(df_1d['close'].values, 50)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(rsi_4h[i]) or np.isnan(atr_4h[i]) or atr_4h[i] <= 1e-10:
            continue
        if np.isnan(ema_1d_aligned[i]) or np.isnan(keltner_ema[i]):
            continue
        if np.isnan(keltner_upper[i]) or np.isnan(keltner_lower[i]):
            continue
        if np.isnan(adx_4h[i]) or np.isnan(chop_4h[i]):
            continue
        if np.isnan(vol_sma_4h[i]) or vol_sma_4h[i] <= 1e-10:
            continue
        
        # === TREND BIAS (1d HTF EMA50) ===
        trend_1d_bullish = close[i] > ema_1d_aligned[i]
        trend_1d_bearish = close[i] < ema_1d_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        trending_regime = chop_4h[i] < 38.2
        ranging_regime = chop_4h[i] > 61.8
        
        # === TREND STRENGTH (ADX) ===
        strong_trend = adx_4h[i] > 25
        weak_trend = adx_4h[i] < 20
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = volume[i] > 1.5 * vol_sma_4h[i]
        
        # === RSI SIGNALS ===
        rsi_oversold = rsi_4h[i] < 30
        rsi_overbought = rsi_4h[i] > 70
        rsi_neutral_long = 40 < rsi_4h[i] < 55
        rsi_neutral_short = 45 < rsi_4h[i] < 60
        
        # === KELTNER POSITION ===
        below_keltner_lower = close[i] < keltner_lower[i]
        above_keltner_upper = close[i] > keltner_upper[i]
        near_keltner_lower = close[i] < keltner_ema[i] and close[i] > keltner_lower[i] * 0.995
        near_keltner_upper = close[i] > keltner_ema[i] and close[i] < keltner_upper[i] * 1.005
        
        desired_signal = 0.0
        
        # === RANGING REGIME LOGIC (CHOP > 61.8) ===
        if ranging_regime:
            # Mean reversion long: RSI oversold + below Keltner lower
            if rsi_oversold and below_keltner_lower and not trend_1d_bearish:
                desired_signal = BASE_SIZE if volume_confirmed else REDUCED_SIZE
            
            # Mean reversion short: RSI overbought + above Keltner upper
            if rsi_overbought and above_keltner_upper and not trend_1d_bullish:
                desired_signal = -BASE_SIZE if volume_confirmed else -REDUCED_SIZE
            
            # Conservative mean reversion: RSI extreme + Keltner touch
            if rsi_oversold and near_keltner_lower and trend_1d_bullish:
                desired_signal = REDUCED_SIZE
            
            if rsi_overbought and near_keltner_upper and trend_1d_bearish:
                desired_signal = -REDUCED_SIZE
        
        # === TRENDING REGIME LOGIC (CHOP < 38.2) ===
        elif trending_regime:
            # Trend pullback long: 1d bullish + RSI neutral + near Keltner EMA
            if trend_1d_bullish and rsi_neutral_long and strong_trend:
                if near_keltner_lower or close[i] < keltner_ema[i]:
                    desired_signal = BASE_SIZE if volume_confirmed else REDUCED_SIZE
            
            # Trend pullback short: 1d bearish + RSI neutral + near Keltner EMA
            if trend_1d_bearish and rsi_neutral_short and strong_trend:
                if near_keltner_upper or close[i] > keltner_ema[i]:
                    desired_signal = -BASE_SIZE if volume_confirmed else -REDUCED_SIZE
            
            # Breakout continuation with volume
            if trend_1d_bullish and above_keltner_upper and volume_confirmed and strong_trend:
                desired_signal = BASE_SIZE
            
            if trend_1d_bearish and below_keltner_lower and volume_confirmed and strong_trend:
                desired_signal = -BASE_SIZE
        
        # === NEUTRAL REGIME (38.2 <= CHOP <= 61.8) ===
        else:
            # Conservative: only enter on RSI extremes + trend alignment
            if rsi_oversold and trend_1d_bullish:
                desired_signal = REDUCED_SIZE
            
            if rsi_overbought and trend_1d_bearish:
                desired_signal = -REDUCED_SIZE
        
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
        
        # === HOLD LOGIC — Maintain position if regime intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if trend intact and not overbought
                if trend_1d_bullish and rsi_4h[i] < 75:
                    desired_signal = BASE_SIZE if trending_regime else REDUCED_SIZE
            elif position_side < 0:
                # Hold short if trend intact and not oversold
                if trend_1d_bearish and rsi_4h[i] > 25:
                    desired_signal = -BASE_SIZE if trending_regime else -REDUCED_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if trend reverses or RSI overbought
            if trend_1d_bearish and rsi_4h[i] > 65:
                desired_signal = 0.0
            # Exit if price hits Keltner upper in ranging regime
            if ranging_regime and above_keltner_upper:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if trend reverses or RSI oversold
            if trend_1d_bullish and rsi_4h[i] < 35:
                desired_signal = 0.0
            # Exit if price hits Keltner lower in ranging regime
            if ranging_regime and below_keltner_lower:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE if desired_signal >= BASE_SIZE else REDUCED_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE if desired_signal <= -BASE_SIZE else -REDUCED_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
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
```

## Last Updated
2026-03-23 14:12
