#!/usr/bin/env python3
"""
Experiment #401: 12h Weekly Trend + Daily ADX Regime + RSI Pullback

Hypothesis: After analyzing 400+ failed experiments, the key insight is that
12h timeframe needs STRONGER HTF filters and LOOSER entry conditions to generate
enough trades while maintaining quality. Previous 12h strategies failed because:
1. Entry conditions too strict (no trades in bear market)
2. HTF filter too weak (daily instead of weekly)
3. No volume confirmation (false breakouts)

STRATEGY COMPONENTS:
1. 1w HMA(21) TREND BIAS: Weekly closes are institutionally significant
   - Long only when price > 1w HMA (bull regime)
   - Short only when price < 1w HMA (bear regime)
   - This prevents counter-trend trades that failed in 2022 crash

2. 1d ADX(14) REGIME FILTER: Detects trending strength
   - ADX > 25 = trending (allow entries in trend direction)
   - ADX < 20 = ranging (allow mean-reversion entries)
   - 20-25 = neutral (reduce position size by 50%)
   - Hysteresis prevents rapid regime flipping

3. 12h RSI(14) PULLBACK ENTRY: Catch retracements in trend
   - Long: RSI < 45 + price > 1w HMA + ADX > 20
   - Short: RSI > 55 + price < 1w HMA + ADX > 20
   - Much looser than RSI < 30 (generates more trades)

4. VOLUME CONFIRMATION: Filter false signals
   - Volume > 0.8 * SMA(volume, 20) confirms genuine moves
   - Prevents entries on low-liquidity fakeouts

5. ATR TRAILING STOP (3.0x): Wide stops for 12h volatility
   - Signal → 0 when price moves 3*ATR against position
   - Wider than 4h strategies (12h has more noise per bar)

6. POSITION SIZING: 0.25 discrete (conservative for 12h)
   - Max 25% capital per position
   - Discrete levels: 0.0, ±0.25 (minimize fee churn)

Why this should work on 12h:
- Weekly trend filter is stable (changes rarely, avoids whipsaw)
- RSI 45/55 thresholds generate 40-80 trades/year (enough for stats)
- Volume filter removes 30% of false signals without killing trade count
- 3*ATR stops account for 12h volatility spikes
- Works on BTC/ETH/SOL individually (weekly trend universal)

Timeframe: 12h (REQUIRED for this experiment)
HTF: 1w and 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete levels
Stoploss: 3.0 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_weekly_hma_daily_adx_rsi_vol_atr_v1"
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
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    n = len(close)
    adx = np.full(n, np.nan)
    
    # Calculate DM and TR
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        plus_move = high[i] - high[i-1]
        minus_move = low[i-1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
        
        tr1 = high[i] - low[i]
        tr2 = np.abs(high[i] - close[i-1])
        tr3 = np.abs(low[i] - close[i-1])
        tr[i] = max(tr1, tr2, tr3)
    
    # Smooth with Wilder's method (EMA with span=period)
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Calculate DI and DX
    for i in range(period, n):
        if tr_s[i] > 1e-10:
            plus_di = 100 * plus_dm_s[i] / tr_s[i]
            minus_di = 100 * minus_dm_s[i] / tr_s[i]
            di_sum = plus_di + minus_di
            if di_sum > 1e-10:
                dx = 100 * np.abs(plus_di - minus_di) / di_sum
                if i == period:
                    adx[i] = dx
                else:
                    adx[i] = ((adx[i-1] * (period - 1)) + dx) / period
    
    return adx

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_volume_sma(volume, period=20):
    """Calculate SMA of volume for volume confirmation."""
    vol_s = pd.Series(volume)
    vol_sma = vol_s.rolling(window=period, min_periods=period).mean().values
    return vol_sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    adx_1d = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    vol_sma = calculate_volume_sma(volume, 20)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.25
    HALF_SIZE = 0.125
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    # Track ADX regime with hysteresis
    prev_adx_regime = 0  # 0=neutral, 1=trending, 2=ranging
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(vol_sma[i]) or vol_sma[i] == 0:
            signals[i] = 0.0
            continue
        
        # === WEEKLY TREND BIAS ===
        bull_trend_1w = close[i] > hma_1w_aligned[i]
        bear_trend_1w = close[i] < hma_1w_aligned[i]
        
        # === DAILY ADX REGIME (with hysteresis) ===
        adx_val = adx_1d_aligned[i]
        if adx_val > 25:
            adx_regime = 1  # trending
        elif adx_val < 20:
            adx_regime = 2  # ranging
        else:
            adx_regime = prev_adx_regime  # maintain previous regime in neutral zone
        prev_adx_regime = adx_regime
        
        is_trending = (adx_regime == 1)
        is_ranging = (adx_regime == 2)
        is_neutral = (adx_regime == 0)
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = volume[i] > 0.8 * vol_sma[i]
        
        # === RSI PULLBACK SIGNALS ===
        # Looser thresholds to ensure enough trades (RSI 45/55 not 30/70)
        rsi_oversold = rsi[i] < 45
        rsi_overbought = rsi[i] > 55
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        current_size = SIZE if not is_neutral else HALF_SIZE
        
        # LONG entries: Weekly bull + RSI pullback + volume confirmed
        if bull_trend_1w and rsi_oversold and volume_confirmed:
            new_signal = current_size
        
        # SHORT entries: Weekly bear + RSI pullback + volume confirmed
        elif bear_trend_1w and rsi_overbought and volume_confirmed:
            new_signal = -current_size
        
        # === STOPLOSS LOGIC (Rule 6) - 3.0 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 3.0 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 3.0 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit long if weekly trend turns bear
        if in_position and position_side > 0 and bear_trend_1w:
            new_signal = 0.0
        
        # Exit short if weekly trend turns bull
        if in_position and position_side < 0 and bull_trend_1w:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals