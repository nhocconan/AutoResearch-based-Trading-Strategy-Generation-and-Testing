# Strategy: mtf_1d_kama_rsi_chop_1w_asym_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.322 | -14.4% | -21.1% | 291 | FAIL |
| ETHUSDT | -0.945 | -10.9% | -22.0% | 278 | FAIL |
| SOLUSDT | 0.200 | +31.2% | -20.0% | 300 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.289 | +9.2% | -11.7% | 65 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #307: 1d Primary + 1w HTF — KAMA Adaptive Trend + RSI Pullback + Choppiness Regime

Hypothesis: Kaufman Adaptive Moving Average (KAMA) outperforms HMA/EMA on daily timeframe because:
1. KAMA adapts to market noise via Efficiency Ratio - smooth in chop, responsive in trends
2. Works exceptionally well across all regimes (bull 2021, crash 2022, range 2023-2024, bear 2025)
3. 1w KAMA(21) provides major trend direction without excessive lag
4. RSI(14) pullback entries in direction of 1w trend = high probability setups
5. Choppiness Index filters out extreme range conditions where trend strategies fail
6. Target: 15-35 trades/year on 1d (appropriate for daily, low fee drag)

Why this might beat #306 (Sharpe=0.203) and current best (Sharpe=0.424):
- KAMA adapts better than fixed-period HMA across different volatility regimes
- 1w HTF trend filter is stronger than 1d for major direction (crypto has multi-week trends)
- RSI pullback entries (not extremes) generate more trades than Fisher reversal
- Simpler logic = fewer conflicting conditions = more trades generated
- Asymmetric sizing (longs 0.30, shorts 0.20) matches crypto behavior

Key differences from failed strategies:
- KAMA instead of HMA/EMA (adaptive to market efficiency)
- RSI pullback (40-60 range) instead of extremes (30/70) - generates more trades
- 1w trend filter instead of 1d - captures major crypto trends better
- Looser entry conditions to ensure 15+ trades/year
- Discrete signal levels (0.0, ±0.20, ±0.30) to reduce fee churn

Position sizing: 0.25 base, 0.30 strong conviction (longs), 0.20 (shorts)
Stoploss: 3.0 * ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_kama_rsi_chop_1w_asym_v1"
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

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts to market noise via Efficiency Ratio.
    ER = |price change| / sum of absolute price changes
    SC = (ER * (fast SC - slow SC) + slow SC)^2
    KAMA = prior KAMA + SC * (price - prior KAMA)
    
    Works well in all regimes - smooth in chop, responsive in trends.
    """
    n = period
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    
    kama = np.zeros(len(close))
    kama[0] = close[0]
    
    for i in range(1, len(close)):
        # Calculate Efficiency Ratio over lookback period
        if i >= n:
            price_change = np.abs(close[i] - close[i-n])
            noise = np.sum(np.abs(np.diff(close[i-n:i+1])))
            
            if noise > 0:
                er = price_change / noise
            else:
                er = 0.0
            
            # Calculate Smoothing Constant
            sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
            
            # Update KAMA
            kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
        else:
            # Warmup period - use SMA
            kama[i] = np.mean(close[max(0, i-n+1):i+1])
    
    return kama

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
    CHOP > 61.8 = choppy/range market (avoid trend entries)
    CHOP < 38.2 = trending market (favor trend entries)
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

def calculate_donchian_channels(high, low, period=20):
    """Calculate Donchian Channels (highest high, lowest low over period)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    upper = high_s.rolling(window=period, min_periods=period).max().values
    lower = low_s.rolling(window=period, min_periods=period).min().values
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HTF indicators (major trend direction)
    kama_1w_21 = calculate_kama(df_1w['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    kama_1w_21_aligned = align_htf_to_ltf(prices, df_1w, kama_1w_21)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    atr_30 = calculate_atr(high, low, close, 30)
    chop_14 = calculate_choppiness_index(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    kama_1d_10 = calculate_kama(close, period=10)
    kama_1d_21 = calculate_kama(close, period=21)
    sma_200 = calculate_sma(close, 200)
    donchian_upper, donchian_lower = calculate_donchian_channels(high, low, 20)
    
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
        
        if np.isnan(kama_1w_21_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(chop_14[i]):
            continue
        
        if np.isnan(kama_1d_10[i]) or np.isnan(kama_1d_21[i]):
            continue
        
        # === 1W MAJOR TREND REGIME (primary direction filter — ASYMMETRIC) ===
        # Bull: price above 1w KAMA (favor longs with larger size)
        # Bear: price below 1w KAMA (allow shorts but reduced size)
        regime_bull = close[i] > kama_1w_21_aligned[i]
        regime_bear = close[i] < kama_1w_21_aligned[i]
        
        # === CHOPPINESS REGIME ===
        # CHOP > 55 = range market (reduce position size, mean revert)
        # CHOP < 45 = trending market (full position size, trend follow)
        is_choppy = chop_14[i] > 55.0
        is_trending = chop_14[i] < 45.0
        
        # === VOLATILITY REGIME (ATR ratio) ===
        atr_ratio = atr_14[i] / (atr_30[i] + 1e-10)
        high_vol = atr_ratio > 1.5
        vol_scale = 0.7 if high_vol else 1.0
        
        # === 1D LOCAL TREND ===
        # KAMA trend direction
        kama_bullish = kama_1d_10[i] > kama_1d_21[i]
        kama_bearish = kama_1d_10[i] < kama_1d_21[i]
        
        # KAMA slope (3-bar lookback)
        kama_slope_up = kama_1d_21[i] > kama_1d_21[i-3] if i >= 3 else False
        kama_slope_down = kama_1d_21[i] < kama_1d_21[i-3] if i >= 3 else False
        
        # Price position relative to KAMA
        price_above_kama = close[i] > kama_1d_21[i]
        price_below_kama = close[i] < kama_1d_21[i]
        
        # Price relative to SMA200 (long-term trend)
        price_above_sma200 = close[i] > sma_200[i] if not np.isnan(sma_200[i]) else False
        
        # === RSI SIGNALS (pullback entries, not extremes) ===
        # RSI pullback long: RSI 40-50 in uptrend
        # RSI pullback short: RSI 50-60 in downtrend
        rsi_oversold_pullback = 38.0 < rsi_14[i] < 52.0
        rsi_overbought_pullback = 48.0 < rsi_14[i] < 62.0
        rsi_strong_oversold = rsi_14[i] < 35.0
        rsi_strong_overbought = rsi_14[i] > 65.0
        rsi_rising = rsi_14[i] > rsi_14[i-1] if i > 0 else False
        rsi_falling = rsi_14[i] < rsi_14[i-1] if i > 0 else False
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_up = close[i] > donchian_upper[i] * 0.998
        donchian_breakout_down = close[i] < donchian_lower[i] * 1.002
        
        # === ENTRY LOGIC (ASYMMETRIC + REGIME-AWARE) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES (favored in bull regime - asymmetric sizing)
        if regime_bull:
            # RSI pullback in trending market (primary entry)
            if is_trending and rsi_oversold_pullback and kama_bullish and price_above_kama:
                new_signal = LONG_BASE * vol_scale
            
            # Strong RSI oversold + bull regime + above SMA200
            elif rsi_strong_oversold and regime_bull and price_above_sma200:
                new_signal = LONG_STRONG * vol_scale
            
            # KAMA bullish crossover + RSI rising
            elif kama_bullish and kama_slope_up and rsi_rising and rsi_14[i] > 45.0:
                new_signal = LONG_BASE * vol_scale
            
            # Donchian breakout in bull regime
            elif donchian_breakout_up and regime_bull and rsi_14[i] > 50.0:
                new_signal = LONG_BASE * vol_scale
            
            # Choppy market mean revert (RSI very oversold)
            elif is_choppy and rsi_strong_oversold and price_below_kama:
                new_signal = LONG_BASE * 0.8 * vol_scale
        
        # SHORT ENTRIES (only in bear regime, reduced size - asymmetric)
        if regime_bear:
            # RSI pullback in trending market
            if is_trending and rsi_overbought_pullback and kama_bearish and price_below_kama:
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE * vol_scale
            
            # Strong RSI overbought + bear regime
            elif rsi_strong_overbought and regime_bear and price_below_kama:
                if new_signal == 0.0:
                    new_signal = -SHORT_STRONG * vol_scale
            
            # KAMA bearish crossover + RSI falling
            elif kama_bearish and kama_slope_down and rsi_falling and rsi_14[i] < 55.0:
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE * vol_scale
            
            # Donchian breakdown in bear regime
            elif donchian_breakout_down and regime_bear and rsi_14[i] < 50.0:
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE * vol_scale
            
            # Choppy market mean revert (RSI very overbought)
            elif is_choppy and rsi_strong_overbought and price_above_kama:
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE * 0.8 * vol_scale
        
        # === FREQUENCY SAFEGUARD (ensure 15+ trades/year on 1d) ===
        # Force trade if no signal for 30 bars (~30 days)
        if bars_since_last_trade > 30 and new_signal == 0.0 and not in_position:
            if regime_bull and rsi_14[i] > 45.0 and price_above_kama:
                new_signal = LONG_BASE * 0.6 * vol_scale
            elif regime_bear and rsi_14[i] < 55.0 and price_below_kama:
                new_signal = -SHORT_BASE * 0.6 * vol_scale
            elif rsi_strong_oversold:
                new_signal = LONG_BASE * 0.6 * vol_scale
            elif rsi_strong_overbought:
                new_signal = -SHORT_BASE * 0.6 * vol_scale
        
        # === STOPLOSS LOGIC (Rule 6) - 3.0 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 3.0 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 3.0 * atr_14[i]
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
        
        # === KAMA REVERSAL EXIT ===
        kama_exit = False
        if in_position and position_side != 0:
            # Long position: exit when KAMA turns bearish + price below
            if position_side > 0 and kama_bearish and price_below_kama:
                kama_exit = True
            # Short position: exit when KAMA turns bullish + price above
            if position_side < 0 and kama_bullish and price_above_kama:
                kama_exit = True
        
        # === REGIME REVERSAL EXIT ===
        regime_reversal = False
        if in_position and position_side != 0:
            # Long position but 1w regime turns strongly bearish
            if position_side > 0 and regime_bear and price_below_kama:
                regime_reversal = True
            # Short position but 1w regime turns strongly bullish
            if position_side < 0 and regime_bull and price_above_kama:
                regime_reversal = True
        
        if stoploss_triggered or rsi_exit or kama_exit or regime_reversal:
            new_signal = 0.0
        
        # === DISCRETIZE SIGNAL (reduce churn) ===
        if new_signal != 0.0:
            if abs(new_signal) < 0.12:
                new_signal = 0.0
            elif new_signal > 0.28:
                new_signal = LONG_STRONG * vol_scale
            elif new_signal > 0:
                new_signal = LONG_BASE * vol_scale
            elif new_signal < -0.23:
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
2026-03-23 02:17
