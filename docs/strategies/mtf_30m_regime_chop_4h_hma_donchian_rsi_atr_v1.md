# Strategy: mtf_30m_regime_chop_4h_hma_donchian_rsi_atr_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | -0.396 | -11.1% | -36.6% | 568 | FAIL |
| SOLUSDT | 0.121 | +20.8% | -36.8% | 397 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.550 | +17.9% | -11.7% | 136 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #014: 30m Regime-Adaptive Strategy with 4h HTF Trend Filter
Hypothesis: 30m timeframe balances noise reduction with trade frequency.
Using Choppiness Index (CHOP) to detect regime: CHOP>61.8=range(mean revert),
CHOP<38.2=trend(breakout). 4h HMA for primary trend bias. Multiple entry paths
ensure >=10 trades per symbol. Conservative sizing (0.25-0.30) with 2.0*ATR stop.
Timeframe: 30m (REQUIRED), HTF: 4h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_regime_chop_4h_hma_donchian_rsi_atr_v1"
timeframe = "30m"
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
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_rsi(close, period=14):
    """Calculate RSI indicator."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    rs = np.where(avg_l > 0, avg_g / avg_l, 100.0)
    rsi = 100 - 100 / (1 + rs)
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_zscore(close, period=20):
    """Calculate Z-score of price relative to rolling mean."""
    close_s = pd.Series(close)
    mean = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    zscore = (close - mean) / (std + 1e-10)
    return zscore

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = ranging market (mean reversion)
    CHOP < 38.2 = trending market (trend following)
    """
    n = len(close)
    chop = np.zeros(n)
    chop[:] = np.nan
    
    atr = calculate_atr(high, low, close, period)
    
    for i in range(period, n):
        atr_sum = np.sum(atr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 0 and atr_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high, lowest low over period)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    mid = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    return upper, mid, lower

def calculate_volume_ratio(taker_buy_volume, volume):
    """Calculate taker buy volume ratio (0-1, >0.5 = bullish pressure)."""
    ratio = np.zeros(len(volume))
    mask = volume > 0
    ratio[mask] = taker_buy_volume[mask] / volume[mask]
    return ratio

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    n = len(close)
    adx = np.zeros(n)
    adx[:] = np.nan
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    atr = calculate_atr(high, low, close, period)
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(period, n):
        if atr[i] > 0:
            plus_di[i] = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values[i] / atr[i]
            minus_di[i] = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values[i] / atr[i]
    
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx[period:] = pd.Series(dx[period:]).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_vol = prices["taker_buy_volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    rsi_fast = calculate_rsi(close, 7)
    zscore = calculate_zscore(close, 20)
    adx = calculate_adx(high, low, close, 14)
    chop = calculate_choppiness(high, low, close, 14)
    vol_ratio = calculate_volume_ratio(taker_buy_vol, volume)
    
    # Donchian Channel
    donch_upper, donch_lower = calculate_donchian(high, low, 20)
    
    # Bollinger Bands
    bb_upper, bb_mid, bb_lower = calculate_bollinger(close, 20, 2.0)
    
    # HMA for 30m trend
    hma_30m = calculate_hma(close, 21)
    hma_30m_fast = calculate_hma(close, 10)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.28
    SIZE_HALF = 0.14
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(zscore[i]) or np.isnan(adx[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(donch_upper[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (HTF)
        htf_bullish = close[i] > hma_4h_aligned[i]
        htf_bearish = close[i] < hma_4h_aligned[i]
        
        # 30m trend
        hma_30m_bullish = close[i] > hma_30m[i]
        hma_30m_bearish = close[i] < hma_30m[i]
        hma_rising = hma_30m[i] > hma_30m[i-1] if i > 0 else False
        hma_falling = hma_30m[i] < hma_30m[i-1] if i > 0 else False
        
        # HMA crossover
        fast_above_slow = hma_30m_fast[i] > hma_30m[i]
        fast_below_slow = hma_30m_fast[i] < hma_30m[i]
        
        # Donchian breakout (use previous bar to avoid look-ahead)
        breakout_long = close[i] > donch_upper[i-1] if i > 0 else False
        breakout_short = close[i] < donch_lower[i-1] if i > 0 else False
        
        # Volume confirmation
        vol_bullish = vol_ratio[i] > 0.48
        vol_bearish = vol_ratio[i] < 0.52
        
        # ADX regime
        trend_strong = adx[i] > 20
        trend_weak = adx[i] < 25
        
        # Choppiness Index regime
        chop_range = chop[i] > 55  # ranging market
        chop_trend = chop[i] < 45  # trending market
        
        # Z-score extremes
        zscore_oversold = zscore[i] < -1.2
        zscore_overbought = zscore[i] > 1.2
        
        # RSI extremes
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        rsi_fast_oversold = rsi_fast[i] < 25
        rsi_fast_overbought = rsi_fast[i] > 75
        
        # Bollinger Band position
        price_near_lower = close[i] < bb_lower[i] * 1.005
        price_near_upper = close[i] > bb_upper[i] * 0.995
        
        new_signal = 0.0
        
        # === REGIME-ADAPTIVE LONG ENTRIES ===
        
        # TREND REGIME (CHOP < 45): Use breakout strategies
        if chop_trend:
            # Path 1: Donchian breakout + 4h bullish + volume (primary trend breakout)
            if breakout_long and htf_bullish and vol_bullish:
                new_signal = SIZE_ENTRY
            
            # Path 2: 4h bullish + 30m HMA bullish + Fast HMA crossover up
            elif htf_bullish and hma_30m_bullish and fast_above_slow and hma_rising:
                new_signal = SIZE_ENTRY
            
            # Path 3: ADX strong + 4h bullish + HMA crossover (momentum entry)
            elif trend_strong and htf_bullish and fast_above_slow:
                new_signal = SIZE_ENTRY
        
        # RANGE REGIME (CHOP > 55): Use mean reversion strategies
        elif chop_range:
            # Path 4: Z-score oversold + 4h bullish (mean reversion in uptrend)
            if zscore_oversold and htf_bullish:
                new_signal = SIZE_ENTRY
            
            # Path 5: RSI oversold + 4h bullish + ADX weak (dip buy in ranging uptrend)
            elif rsi_oversold and htf_bullish and trend_weak:
                new_signal = SIZE_ENTRY
            
            # Path 6: Price at BB lower + 4h bullish + RSI oversold (deep pullback)
            elif price_near_lower and htf_bullish and rsi_oversold:
                new_signal = SIZE_ENTRY
            
            # Path 7: RSI fast oversold + zscore oversold (double confirmation)
            elif rsi_fast_oversold and zscore_oversold:
                new_signal = SIZE_ENTRY
        
        # NEUTRAL REGIME (45 <= CHOP <= 55): Mixed approach
        else:
            # Path 8: 4h bullish + HMA bullish + RSI not overbought
            if htf_bullish and hma_30m_bullish and rsi[i] < 70:
                new_signal = SIZE_ENTRY
            
            # Path 9: Breakout with volume confirmation
            elif breakout_long and vol_bullish:
                new_signal = SIZE_ENTRY
        
        # === REGIME-ADAPTIVE SHORT ENTRIES ===
        
        # TREND REGIME (CHOP < 45): Use breakout strategies
        if chop_trend:
            # Path 1: Donchian breakout + 4h bearish + volume
            if breakout_short and htf_bearish and vol_bearish:
                new_signal = -SIZE_ENTRY
            
            # Path 2: 4h bearish + 30m HMA bearish + Fast HMA crossover down
            elif htf_bearish and hma_30m_bearish and fast_below_slow and hma_falling:
                new_signal = -SIZE_ENTRY
            
            # Path 3: ADX strong + 4h bearish + HMA crossover
            elif trend_strong and htf_bearish and fast_below_slow:
                new_signal = -SIZE_ENTRY
        
        # RANGE REGIME (CHOP > 55): Use mean reversion strategies
        elif chop_range:
            # Path 4: Z-score overbought + 4h bearish
            if zscore_overbought and htf_bearish:
                new_signal = -SIZE_ENTRY
            
            # Path 5: RSI overbought + 4h bearish + ADX weak
            elif rsi_overbought and htf_bearish and trend_weak:
                new_signal = -SIZE_ENTRY
            
            # Path 6: Price at BB upper + 4h bearish + RSI overbought
            elif price_near_upper and htf_bearish and rsi_overbought:
                new_signal = -SIZE_ENTRY
            
            # Path 7: RSI fast overbought + zscore overbought
            elif rsi_fast_overbought and zscore_overbought:
                new_signal = -SIZE_ENTRY
        
        # NEUTRAL REGIME (45 <= CHOP <= 55): Mixed approach
        else:
            # Path 8: 4h bearish + HMA bearish + RSI not oversold
            if htf_bearish and hma_30m_bearish and rsi[i] > 30:
                new_signal = -SIZE_ENTRY
            
            # Path 9: Breakout with volume confirmation
            elif breakout_short and vol_bearish:
                new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.0*ATR for 30m timeframe)
            current_stop = highest_close - 2.0 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.0 * atr[i]
                profit = close[i] - entry_price
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.0*ATR for 30m timeframe)
            current_stop = lowest_close + 2.0 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.0 * atr[i]
                profit = entry_price - close[i]
                if profit >= 2.0 * risk:
                    new_signal = -SIZE_HALF
                    position_reduced = True
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.0 * atr[i] if position_side > 0 else close[i] + 2.0 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.0 * atr[i] if position_side > 0 else close[i] + 2.0 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reduced (take profit)
        elif new_signal != 0.0 and prev_signal != 0.0 and np.abs(new_signal) < np.abs(prev_signal):
            position_reduced = True
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
            position_reduced = False
        
        signals[i] = new_signal
    
    return signals
```

## Last Updated
2026-03-22 07:50
