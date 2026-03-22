#!/usr/bin/env python3
"""
Experiment #509: 4h Primary + 1d HTF — KAMA Adaptive Trend + ADX Regime + Asymmetric RSI

Hypothesis: After 498 failed strategies (mostly HMA/EMA/Fisher/VolSpike combos), try 
KAMA (Kaufman Adaptive Moving Average) which ADAPTS to market noise - flattens in chop,
follows trend in trending markets. This is fundamentally DIFFERENT from fixed-period MAs.

Key innovations:
1. KAMA adapts efficiency ratio (ER) based on price movement vs noise
   - High ER (trending): KAMA follows price closely
   - Low ER (choppy): KAMA flattens, avoids whipsaws
   - This should reduce 2022-style crash losses

2. ADX Regime Detection with HYSTERESIS:
   - ADX > 25 = trending (follow KAMA direction)
   - ADX < 20 = ranging (mean revert at RSI extremes)
   - Hysteresis prevents rapid regime flipping (enter 25, exit 18)

3. Asymmetric RSI thresholds based on 1d HTF trend:
   - Bull regime (price > 1d HMA): Long at RSI<40, Short only at RSI>80
   - Bear regime (price < 1d HMA): Short at RSI>60, Long only at RSI<20
   - This creates bias toward major trend while allowing counter-trend at extremes

4. 1d HTF HMA for major trend filter (proven in current best strategy)

Why this might beat current best (Sharpe=0.435):
- KAMA is DIFFERENT from all failed HMA/EMA strategies
- ADX regime filter prevents trend-following in chop (major failure mode)
- Asymmetric RSI generates trades in both regimes (avoids 0-trade problem)
- 4h TF targets 25-45 trades/year (optimal fee/trade ratio)

Position sizing: 0.25-0.30 discrete levels
Stoploss: 2.5 * ATR trailing (signal → 0 when hit)
Target: >=30 trades/symbol on train, >=3 on test, Sharpe > 0.435
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_adx_regime_asym_rsi_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, fast_period=2, slow_period=30, smoothing_period=10):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    Adapts to market noise - follows trend when ER is high, flattens when choppy.
    
    Efficiency Ratio (ER) = |Net Change| / Sum of Absolute Changes
    - ER near 1: strong trend (KAMA follows price)
    - ER near 0: choppy (KAMA flattens)
    
    Smoothing Constant (SC) = [ER * (fast_SC - slow_SC) + slow_SC]^2
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Net change over slow_period
    net_change = np.abs(close_s - close_s.shift(slow_period))
    
    # Sum of absolute changes (volatility/noise)
    abs_changes = np.abs(close_s.diff())
    sum_abs_changes = abs_changes.rolling(window=slow_period, min_periods=slow_period).sum()
    
    # Efficiency Ratio (0 to 1)
    er = net_change / (sum_abs_changes + 1e-10)
    er = er.clip(0, 1)
    
    # Smoothing constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Adaptive smoothing constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation (iterative)
    kama = np.zeros(n)
    kama[slow_period] = close[slow_period]  # Initialize
    
    for i in range(slow_period + 1, n):
        if np.isnan(sc.iloc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_adx(high, low, close, period=14):
    """
    Calculate Average Directional Index (ADX).
    Measures trend strength (not direction).
    - ADX > 25: trending market
    - ADX < 20: ranging/choppy market
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    # Smoothed values (Wilder's smoothing)
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / (atr + 1e-10))
    minus_di = 100 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / (atr + 1e-10))
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values, plus_di.values, minus_di.values

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
    """Calculate Hull Moving Average (HMA) for 1d trend filter."""
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HTF HMA for major trend
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_50 = calculate_hma(df_1d['close'].values, period=50)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_50_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_50)
    
    # Calculate 4h indicators
    kama = calculate_kama(close, fast_period=2, slow_period=30, smoothing_period=10)
    atr_14 = calculate_atr(high, low, close, 14)
    adx, plus_di, minus_di = calculate_adx(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    LONG_SIZE = 0.30
    SHORT_SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # ADX regime hysteresis tracking
    prev_adx_regime = 0  # 0=unknown, 1=trending, 2=ranging
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_50_aligned[i]):
            continue
        if np.isnan(kama[i]) or np.isnan(adx[i]) or np.isnan(rsi_14[i]):
            continue
        
        # === 1D MAJOR TREND (primary direction filter) ===
        bull_regime = close[i] > hma_1d_21_aligned[i]
        bear_regime = close[i] < hma_1d_21_aligned[i]
        
        # 1d HMA slope for trend confirmation
        hma_slope_bull = hma_1d_21_aligned[i] > hma_1d_50_aligned[i]
        hma_slope_bear = hma_1d_21_aligned[i] < hma_1d_50_aligned[i]
        
        # === ADX REGIME DETECTION WITH HYSTERESIS ===
        # Trending: ADX > 25, Ranging: ADX < 20, Hysteresis zone: 20-25
        if adx[i] > 25:
            adx_regime = 1  # Trending
        elif adx[i] < 20:
            adx_regime = 2  # Ranging
        else:
            adx_regime = prev_adx_regime  # Keep previous in hysteresis zone
        
        prev_adx_regime = adx_regime
        
        is_trending = (adx_regime == 1)
        is_ranging = (adx_regime == 2)
        
        # === KAMA TREND DIRECTION ===
        kama_bullish = close[i] > kama[i]
        kama_bearish = close[i] < kama[i]
        
        # KAMA slope (trend momentum)
        kama_slope_bull = kama[i] > kama[i-5] if i >= 5 else False
        kama_slope_bear = kama[i] < kama[i-5] if i >= 5 else False
        
        # === ASYMMETRIC RSI THRESHOLDS ===
        # Bull regime: easier to go long, harder to short
        if bull_regime:
            rsi_long_threshold = 40.0   # Long on pullback
            rsi_short_threshold = 80.0  # Short only at extreme
        else:  # Bear regime
            rsi_long_threshold = 20.0   # Long only at extreme
            rsi_short_threshold = 60.0  # Short on bounce
        
        rsi_oversold = rsi_14[i] < rsi_long_threshold
        rsi_overbought = rsi_14[i] > rsi_short_threshold
        
        # Extreme RSI for counter-trend entries
        rsi_extreme_low = rsi_14[i] < 25.0
        rsi_extreme_high = rsi_14[i] > 75.0
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # TRENDING REGIME (ADX > 25): Follow KAMA direction with RSI pullback
        if is_trending:
            # Long: KAMA bullish + RSI pullback in bull regime
            if kama_bullish and kama_slope_bull and rsi_oversold and bull_regime:
                new_signal = LONG_SIZE
            # Long: KAMA bullish breakout + 1d trend confirmation
            elif kama_bullish and kama_slope_bull and hma_slope_bull and close[i] > kama[i-3]:
                new_signal = LONG_SIZE * 0.8
            # Short: KAMA bearish + RSI bounce in bear regime
            elif kama_bearish and kama_slope_bear and rsi_overbought and bear_regime:
                new_signal = -SHORT_SIZE
            # Short: KAMA bearish breakdown + 1d trend confirmation
            elif kama_bearish and kama_slope_bear and hma_slope_bear and close[i] < kama[i-3]:
                new_signal = -SHORT_SIZE * 0.8
        
        # RANGING REGIME (ADX < 20): Mean revert at RSI extremes
        elif is_ranging:
            # Long: RSI extreme low + price near KAMA (support)
            if rsi_extreme_low and abs(close[i] - kama[i]) < 0.02 * close[i]:
                new_signal = LONG_SIZE * 0.7
            # Long: RSI oversold + bull regime (counter-trend in range)
            elif rsi_oversold and bull_regime:
                new_signal = LONG_SIZE * 0.6
            # Short: RSI extreme high + price near KAMA (resistance)
            elif rsi_extreme_high and abs(close[i] - kama[i]) < 0.02 * close[i]:
                new_signal = -SHORT_SIZE * 0.7
            # Short: RSI overbought + bear regime (counter-trend in range)
            elif rsi_overbought and bear_regime:
                new_signal = -SHORT_SIZE * 0.6
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT CONDITIONS ===
        # Exit long on regime flip or KAMA cross
        if in_position and position_side > 0:
            # KAMA bearish cross
            if kama_bearish and kama_slope_bear:
                new_signal = 0.0
            # 1d regime flip to strong bear
            if bear_regime and hma_slope_bear and adx[i] > 25:
                new_signal = 0.0
            # RSI overbought in ranging market
            if is_ranging and rsi_overbought:
                new_signal = 0.0
        
        # Exit short on regime flip or KAMA cross
        if in_position and position_side < 0:
            # KAMA bullish cross
            if kama_bullish and kama_slope_bull:
                new_signal = 0.0
            # 1d regime flip to strong bull
            if bull_regime and hma_slope_bull and adx[i] > 25:
                new_signal = 0.0
            # RSI oversold in ranging market
            if is_ranging and rsi_oversold:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Flip position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals