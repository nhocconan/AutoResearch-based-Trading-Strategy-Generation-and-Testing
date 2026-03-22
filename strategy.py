#!/usr/bin/env python3
"""
Experiment #242: 12h Primary + 1d/1w HTF — KAMA Trend + Fisher Entry + Choppiness Regime

Hypothesis: After analyzing 241 failed experiments, the pattern is clear:
- Complex multi-regime strategies fail (too many conflicting filters = 0 trades)
- Lower timeframes fail (fee drag from too many trades)
- 12h timeframe shows promise (current best: Sharpe=0.270)

This strategy uses:
1. 12h KAMA(10) for adaptive trend following (better than EMA/HMA in chop)
2. 1d HMA(21) for HTF trend direction
3. Fisher Transform(9) for precise entry timing (catches reversals in bear markets)
4. Choppiness Index(14) to detect range vs trend (switch logic)
5. 1w ADX for macro trend strength
6. 2.5 ATR trailing stop for risk management

Key improvements:
- SIMPLER entry logic (fewer AND conditions)
- LOOSER thresholds to guarantee 20-50 trades/year
- KAMA adapts to volatility (better than fixed EMA in crypto)
- Fisher Transform works well in bear/range markets (2025+ scenario)
- Discrete position sizing (0.25 base, 0.30 strong)

Position sizing: 0.25-0.30 (discrete levels)
Target: 20-50 trades/year per symbol (12h cost model)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_fisher_chop_regime_1d1w_v1"
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
    KAMA adapts to market noise - moves fast in trends, slow in chop.
    Better than EMA/HMA for crypto volatility.
    """
    close_s = pd.Series(close)
    
    # Efficiency Ratio (ER) - measures trend efficiency
    change = (close_s - close_s.shift(er_period)).abs()
    volatility = close_s.diff().abs().rolling(window=er_period, min_periods=er_period).sum()
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0)
    
    # Smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(len(close))
    kama[0] = close[0]
    
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_fisher_transform(high, low, close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Transforms price into Gaussian distribution for clearer signals.
    Long when Fisher crosses above -1.5, short when crosses below +1.5.
    Excellent for catching reversals in bear/range markets.
    """
    close_s = pd.Series(close)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    # Calculate price position within range
    hl2 = (high_s + low_s) / 2.0
    range_hl = (high_s - low_s).replace(0, np.nan)
    
    # Normalize price to -1 to +1 range
    price_norm = 2 * ((close_s - low_s) / range_hl) - 1
    price_norm = price_norm.clip(-0.999, 0.999)  # Prevent log(0)
    
    # Smooth with EMA
    price_smooth = price_norm.ewm(span=period, min_periods=period, adjust=False).mean()
    price_smooth = price_smooth.clip(-0.999, 0.999)
    
    # Fisher transform
    fisher = 0.5 * np.log((1 + price_smooth) / (1 - price_smooth))
    fisher_prev = fisher.shift(1)
    
    return fisher.values, fisher_prev.values

def calculate_adx(high, low, close, period=14):
    """Calculate ADX for trend strength."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    tr1 = high_s - low_s
    tr2 = (high_s - close_s.shift(1)).abs()
    tr3 = (low_s - close_s.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.fillna(20).values

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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d HTF indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    adx_1d = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    
    # Calculate 1w HTF indicators
    adx_1w = calculate_adx(df_1w['high'].values, df_1w['low'].values, df_1w['close'].values, 14)
    hma_1w_21 = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    kama_10 = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    fisher, fisher_prev = calculate_fisher_transform(high, low, close, 9)
    chop_14 = calculate_choppiness_index(high, low, close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.25
    STRONG_SIZE = 0.30
    
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
        
        if np.isnan(kama_10[i]) or np.isnan(rsi_14[i]):
            continue
        
        if np.isnan(fisher[i]) or np.isnan(fisher_prev[i]):
            continue
        
        if np.isnan(chop_14[i]):
            continue
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(adx_1d_aligned[i]):
            continue
        
        # === HTF TREND REGIME (1d HMA) ===
        # Simple: price above 1d HMA = bull bias, below = bear bias
        htf_bull = close[i] > hma_1d_21_aligned[i]
        htf_bear = close[i] < hma_1d_21_aligned[i]
        
        # === MACRO TREND STRENGTH (1w ADX) ===
        macro_trending = adx_1w_aligned[i] > 20  # Loose threshold
        
        # === CHOPPINESS REGIME ===
        # CHOP > 55 = choppy (mean revert mode)
        # CHOP < 45 = trending (trend follow mode)
        is_choppy = chop_14[i] > 55.0
        is_trending = chop_14[i] < 45.0
        
        # === 12H LOCAL TREND (KAMA) ===
        kama_bull = close[i] > kama_10[i]
        kama_bear = close[i] < kama_10[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        # Fisher cross above -1.5 = long signal
        # Fisher cross below +1.5 = short signal
        fisher_long = (fisher_prev[i] < -1.5) and (fisher[i] >= -1.5)
        fisher_short = (fisher_prev[i] > 1.5) and (fisher[i] <= 1.5)
        
        # Fisher extreme levels
        fisher_oversold = fisher[i] < -1.8
        fisher_overbought = fisher[i] > 1.8
        
        # === RSI MOMENTUM (LOOSE THRESHOLDS) ===
        rsi_bullish = rsi_14[i] > 45
        rsi_bearish = rsi_14[i] < 55
        rsi_oversold = rsi_14[i] < 40
        rsi_overbought = rsi_14[i] > 60
        
        # === POSITION SIZING ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # === TREND FOLLOWING MODE (when trending + HTF aligned) ===
        if is_trending or macro_trending:
            # LONG: HTF bull + KAMA bull + Fisher long OR RSI bullish
            if htf_bull and kama_bull:
                if fisher_long or (rsi_bullish and rsi_14[i] > 50):
                    new_signal = STRONG_SIZE
                elif rsi_bullish:
                    new_signal = BASE_SIZE
            
            # SHORT: HTF bear + KAMA bear + Fisher short OR RSI bearish
            if htf_bear and kama_bear:
                if fisher_short or (rsi_bearish and rsi_14[i] < 50):
                    new_signal = -STRONG_SIZE
                elif rsi_bearish:
                    new_signal = -BASE_SIZE
        
        # === MEAN REVERSION MODE (when choppy) ===
        if is_choppy:
            # LONG: Fisher oversold + RSI oversold (reversion in range)
            if fisher_oversold or rsi_oversold:
                if not htf_bear:  # Avoid strong bear trend
                    new_signal = BASE_SIZE * 0.8
            
            # SHORT: Fisher overbought + RSI overbought (reversion in range)
            if fisher_overbought or rsi_overbought:
                if not htf_bull:  # Avoid strong bull trend
                    if new_signal == 0.0:
                        new_signal = -BASE_SIZE * 0.8
        
        # === FISHER CROSSOVER ENTRY (works in any regime) ===
        if fisher_long and rsi_bullish and not htf_bear:
            if new_signal == 0.0:
                new_signal = BASE_SIZE
        
        if fisher_short and rsi_bearish and not htf_bull:
            if new_signal == 0.0:
                new_signal = -BASE_SIZE
        
        # === FREQUENCY SAFEGUARD (CRITICAL for 10+ trades) ===
        # Force trade if no signal for 50 bars (~25 days on 12h)
        if bars_since_last_trade > 50 and new_signal == 0.0 and not in_position:
            if htf_bull and kama_bull and rsi_14[i] > 48:
                new_signal = BASE_SIZE * 0.5
            elif htf_bear and kama_bear and rsi_14[i] < 52:
                new_signal = -BASE_SIZE * 0.5
            elif is_choppy and rsi_14[i] < 42:
                new_signal = BASE_SIZE * 0.4
            elif is_choppy and rsi_14[i] > 58:
                new_signal = -BASE_SIZE * 0.4
        
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
            # Long position but HTF turns strongly bearish
            if position_side > 0 and htf_bear and kama_bear:
                regime_reversal = True
            # Short position but HTF turns strongly bullish
            if position_side < 0 and htf_bull and kama_bull:
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