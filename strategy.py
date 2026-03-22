#!/usr/bin/env python3
"""
Experiment #263: 1d Primary + 1w HTF — Dual Regime with HMA Trend Filter

Hypothesis: After analyzing 262 experiments, the winning pattern is clear:
1. 12h/1d primary timeframes work best (less noise, fewer whipsaws)
2. Dual regime (trend vs mean-revert) adapts to market conditions
3. HTF (1w) HMA provides strong trend filter without overfitting
4. Donchian breakouts for trend entries, RSI extremes for mean-revert
5. Choppiness Index cleanly separates regimes

Key differences from failed attempts:
- Simpler than #254/#259 (no complex CRSI, no Fisher transform)
- 1d primary = naturally fewer trades (20-40/year target)
- 1w HTF = stronger trend filter than 1d (used in #262 with Sharpe=0.244)
- Relaxed RSI thresholds to ensure 10+ trades per symbol

Position sizing: 0.30 base, 0.35 strong conviction (discrete levels)
Target: 20-40 trades/year per symbol (appropriate for 1d)
Stoploss: 2.5 * ATR trailing

Lessons from failures:
- #255 Fisher transform: Sharpe=-27 (too noisy)
- #258/#260 Volume-based: Sharpe negative (volume unreliable on daily)
- #254 CRSI: Sharpe=-0.416 (overfitting)
- #256/#262 Dual regime: Positive Sharpe (this works!)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_dual_regime_hma_donchian_rsi_1w_v1"
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

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = choppy/range market (mean revert)
    CHOP < 38.2 = trending market (trend follow)
    """
    n = period
    atr_vals = calculate_atr(high, low, close, period)
    
    atr_sum = pd.Series(atr_vals).rolling(window=n, min_periods=n).sum().values
    hh = pd.Series(high).rolling(window=n, min_periods=n).max().values
    ll = pd.Series(low).rolling(window=n, min_periods=n).min().values
    
    chop = np.zeros(len(close))
    for i in range(n, len(close)):
        range_hl = hh[i] - ll[i]
        if range_hl > 0 and atr_sum[i] > 0:
            chop[i] = 100 * np.log10(atr_sum[i] / range_hl) / np.log10(n)
        else:
            chop[i] = 50.0
    
    return chop

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
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HTF indicators (primary trend regime)
    hma_1w_21 = calculate_hma(df_1w['close'].values, 21)
    hma_1w_50 = calculate_hma(df_1w['close'].values, 50)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    hma_1w_50_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_50)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    adx_14 = calculate_adx(high, low, close, 14)
    chop_14 = calculate_choppiness_index(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    hma_1d_21 = calculate_hma(close, 21)
    hma_1d_50 = calculate_hma(close, 50)
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
    last_trade_bar = -20
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1w_21_aligned[i]) or np.isnan(hma_1w_50_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(hma_1d_21[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(adx_14[i]):
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        # === 1W TREND REGIME (primary direction filter) ===
        # Strong bull: price above both 1w HMA21 and HMA50
        # Strong bear: price below both 1w HMA21 and HMA50
        # Neutral: between HMAs
        regime_strong_bull = close[i] > hma_1w_21_aligned[i] and hma_1w_21_aligned[i] > hma_1w_50_aligned[i]
        regime_strong_bear = close[i] < hma_1w_21_aligned[i] and hma_1w_21_aligned[i] < hma_1w_50_aligned[i]
        regime_bull = close[i] > hma_1w_21_aligned[i]
        regime_bear = close[i] < hma_1w_21_aligned[i]
        
        # === CHOPPINESS REGIME ===
        # CHOP > 55 = range market (mean revert entries)
        # CHOP < 45 = trend market (breakout entries)
        is_choppy = chop_14[i] > 55.0
        is_trending = chop_14[i] < 45.0
        
        # === TREND STRENGTH ===
        is_strong_trend = adx_14[i] > 22.0
        is_weak_trend = adx_14[i] < 18.0
        
        # === 1D LOCAL SIGNALS ===
        price_above_1d_hma = close[i] > hma_1d_21[i]
        price_below_1d_hma = close[i] < hma_1d_21[i]
        hma_1d_bullish = hma_1d_21[i] > hma_1d_50[i]
        hma_1d_bearish = hma_1d_21[i] < hma_1d_50[i]
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] >= donchian_upper[i] * 0.995
        donchian_breakout_short = close[i] <= donchian_lower[i] * 1.005
        
        # === RSI THRESHOLDS (relaxed for more trades on 1d) ===
        rsi_oversold = rsi_14[i] < 40.0
        rsi_overbought = rsi_14[i] > 60.0
        rsi_extreme_oversold = rsi_14[i] < 30.0
        rsi_extreme_overbought = rsi_14[i] > 70.0
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # TREND FOLLOWING MODE (when trending + strong ADX + regime aligned)
        if is_trending and is_strong_trend:
            # LONG: Trending + bull regime + price above 1d HMA + RSI confirming
            if regime_strong_bull and price_above_1d_hma and rsi_14[i] > 45:
                new_signal = STRONG_SIZE
            # LONG: Trending + Donchian breakout + 1d HMA bullish + any bull regime
            elif donchian_breakout_long and hma_1d_bullish and regime_bull:
                new_signal = BASE_SIZE
            
            # SHORT: Trending + bear regime + price below 1d HMA + RSI confirming
            if regime_strong_bear and price_below_1d_hma and rsi_14[i] < 55:
                if new_signal == 0.0 or abs(new_signal) < STRONG_SIZE:
                    new_signal = -STRONG_SIZE
            # SHORT: Trending + Donchian breakdown + 1d HMA bearish + any bear regime
            elif donchian_breakout_short and hma_1d_bearish and regime_bear:
                if new_signal == 0.0:
                    new_signal = -BASE_SIZE
        
        # MEAN REVERSION MODE (when choppy + weak ADX)
        if is_choppy or is_weak_trend:
            # LONG: Choppy + RSI oversold (<40) + not in strong bear
            if rsi_oversold and not regime_strong_bear:
                new_signal = BASE_SIZE
            # LONG: Choppy + RSI extreme oversold (<30) in any regime
            if rsi_extreme_oversold:
                if new_signal == 0.0:
                    new_signal = BASE_SIZE
            
            # SHORT: Choppy + RSI overbought (>60) + not in strong bull
            if rsi_overbought and not regime_strong_bull:
                if new_signal == 0.0:
                    new_signal = -BASE_SIZE
            # SHORT: Choppy + RSI extreme overbought (>70) in any regime
            if rsi_extreme_overbought:
                if new_signal == 0.0:
                    new_signal = -BASE_SIZE
        
        # === FREQUENCY SAFEGUARD (CRITICAL for 10+ trades) ===
        # Force trade if no signal for 15 bars (~15 days on 1d)
        if bars_since_last_trade > 15 and new_signal == 0.0 and not in_position:
            if regime_bull and rsi_14[i] > 40 and price_above_1d_hma:
                new_signal = BASE_SIZE * 0.8
            elif regime_bear and rsi_14[i] < 60 and price_below_1d_hma:
                new_signal = -BASE_SIZE * 0.8
            elif is_choppy and rsi_14[i] < 35:
                new_signal = BASE_SIZE * 0.7
            elif is_choppy and rsi_14[i] > 65:
                new_signal = -BASE_SIZE * 0.7
        
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
            # Long position but regime turns strongly bearish
            if position_side > 0 and regime_strong_bear and price_below_1d_hma:
                regime_reversal = True
            # Short position but regime turns strongly bullish
            if position_side < 0 and regime_strong_bull and price_above_1d_hma:
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