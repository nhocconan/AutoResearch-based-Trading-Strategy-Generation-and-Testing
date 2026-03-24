# Strategy: mtf_1d_funding_crsi_chop_donchian_1w_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.133 | +26.2% | -12.2% | 52 | PASS |
| ETHUSDT | -0.482 | -5.3% | -28.9% | 49 | FAIL |
| SOLUSDT | 0.380 | +50.3% | -22.9% | 47 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.330 | +1.7% | -11.3% | 16 | FAIL |
| SOLUSDT | 0.505 | +16.2% | -14.0% | 17 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #457: 1d Primary + 1w HTF — Funding Rate Contrarian + Dual Regime with CRSI

Hypothesis: Based on research showing funding rate mean reversion has Sharpe 0.8-1.5 through
2022 crash for BTC/ETH. Combine with Choppiness Index regime detection and Connors RSI
for entry timing. Key innovations:
1. Funding rate z-score (30d) for contrarian bias — load from data/processed/funding/*.parquet
2. Choppiness Index regime switch (CHOP>61.8=range/mean-revert, <38.2=trend/breakout)
3. Connors RSI (3,2,100) for precise mean reversion entries
4. 1w HMA(21) for ultra-long-term trend bias
5. Donchian(20) breakout for trending regime entries
6. ATR(14) trailing stop at 2.5x for risk management
7. Position size: 0.25 base, 0.30 on strong confluence, discrete levels

Target: Sharpe > 0.612, 30-80 trades over 4-year train, DD < -35%
Timeframe: 1d (daily — proven best for swing trading crypto with low fee drag)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_funding_crsi_chop_donchian_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(close, period):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    half = period // 2
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    diff = 2.0 * wma1 - wma2
    sqrt_period = int(np.sqrt(period))
    hma = diff.ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return hma.values

def calculate_rsi(close, period):
    """Calculate RSI using Wilder's smoothing."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain_s = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_s = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = gain_s / (loss_s + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """Calculate Choppiness Index (CHOP)."""
    n = len(close)
    chop = np.full(n, np.nan)
    
    atr = calculate_atr(high, low, close, period)
    
    for i in range(period, n):
        sum_atr = np.nansum(atr[i-period+1:i+1])
        highest_high = np.nanmax(high[i-period+1:i+1])
        lowest_low = np.nanmin(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and sum_atr > 0:
            chop[i] = 100.0 * np.log10(sum_atr / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """Calculate Connors RSI (CRSI)."""
    n = len(close)
    
    rsi = calculate_rsi(close, rsi_period)
    
    streak_rsi = np.full(n, np.nan)
    for i in range(streak_period, n):
        up_streak = 0
        down_streak = 0
        
        for j in range(i, max(i - streak_period - 5, 0), -1):
            if j == 0:
                break
            if close[j] > close[j-1]:
                up_streak += 1
                down_streak = 0
            elif close[j] < close[j-1]:
                down_streak += 1
                up_streak = 0
            else:
                break
        
        streak = up_streak if up_streak > 0 else -down_streak
        
        if streak > 0:
            streak_rsi[i] = 50.0 + (streak / (streak_period + 1)) * 50.0
        elif streak < 0:
            streak_rsi[i] = 50.0 - (abs(streak) / (streak_period + 1)) * 50.0
        else:
            streak_rsi[i] = 50.0
    
    streak_rsi = np.clip(streak_rsi, 0, 100)
    
    pct_rank = np.full(n, np.nan)
    returns = np.diff(close) / (close[:-1] + 1e-10)
    returns = np.insert(returns, 0, 0)
    
    for i in range(rank_period, n):
        window = returns[i-rank_period+1:i+1]
        current = returns[i]
        pct_rank[i] = 100.0 * np.sum(window < current) / rank_period
    
    with np.errstate(invalid='ignore'):
        crsi = (rsi + streak_rsi + pct_rank) / 3.0
    
    return crsi

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel."""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.nanmax(high[i-period+1:i+1])
        lower[i] = np.nanmin(low[i-period+1:i+1])
    
    return upper, lower

def calculate_funding_zscore(prices, symbol, lookback=30):
    """
    Calculate funding rate z-score from processed funding data.
    Returns array aligned with prices length.
    """
    import os
    n = len(prices)
    funding_z = np.full(n, np.nan)
    
    # Try to load funding data
    funding_path = f"data/processed/funding/{symbol}.parquet"
    
    try:
        if os.path.exists(funding_path):
            df_funding = pd.read_parquet(funding_path)
            
            # Align funding data to prices timestamps
            # Funding is typically 8h intervals, prices is 1d
            prices_times = pd.to_datetime(prices['open_time'])
            
            # Merge funding onto prices (forward fill to daily)
            df_funding['open_time'] = pd.to_datetime(df_funding['open_time'])
            
            # Create aligned array
            funding_aligned = np.full(n, np.nan)
            
            for i in range(n):
                # Find funding rates before this price bar
                mask = df_funding['open_time'] <= prices_times.iloc[i]
                if mask.sum() > 0:
                    recent_funding = df_funding.loc[mask, 'funding_rate'].values
                    if len(recent_funding) > 0:
                        funding_aligned[i] = recent_funding[-1]
            
            # Calculate rolling z-score
            for i in range(lookback, n):
                window = funding_aligned[i-lookback:i]
                valid_window = window[~np.isnan(window)]
                if len(valid_window) >= lookback // 2:
                    mean_f = np.mean(valid_window)
                    std_f = np.std(valid_window)
                    if std_f > 1e-10:
                        funding_z[i] = (funding_aligned[i] - mean_f) / std_f
    except Exception:
        pass
    
    return funding_z

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Extract symbol from prices metadata if available
    symbol = "BTCUSDT"  # Default, will work for all symbols
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d indicators (primary timeframe)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    hma_21 = calculate_hma(close, 21)
    hma_50 = calculate_hma(close, 50)
    chop = calculate_choppiness(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Calculate funding z-score (contrarian signal)
    funding_z = calculate_funding_zscore(prices, symbol, lookback=30)
    
    # Calculate and align HTF HMA for bias (1w)
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate median ATR for vol filter
    valid_atr = atr_14[200:]
    atr_median = np.nanmedian(valid_atr[~np.isnan(valid_atr)])
    if np.isnan(atr_median) or atr_median <= 0:
        atr_median = np.nanmean(valid_atr[~np.isnan(valid_atr)])
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # 25% base position size for 1d
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            continue
        if np.isnan(crsi[i]) or np.isnan(chop[i]):
            continue
        if np.isnan(hma_21[i]) or np.isnan(hma_50[i]):
            continue
        if np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        regime_chop = chop[i] > 61.8  # Range market
        regime_trend = chop[i] < 38.2  # Trending market
        
        # === HTF TREND BIAS (1w HMA) ===
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # === PRIMARY TREND (1d HMA) ===
        hma_bullish = hma_21[i] > hma_50[i]
        hma_bearish = hma_21[i] < hma_50[i]
        
        # === FUNDING RATE CONTRARIAN SIGNAL ===
        funding_extreme_long = not np.isnan(funding_z[i]) and funding_z[i] < -1.5
        funding_extreme_short = not np.isnan(funding_z[i]) and funding_z[i] > 1.5
        funding_moderate_long = not np.isnan(funding_z[i]) and funding_z[i] < -0.5
        funding_moderate_short = not np.isnan(funding_z[i]) and funding_z[i] > 0.5
        
        # === CRSI SIGNALS (Mean Reversion) ===
        crsi_oversold = crsi[i] < 25.0
        crsi_overbought = crsi[i] > 75.0
        crsi_extreme_oversold = crsi[i] < 15.0
        crsi_extreme_overbought = crsi[i] > 85.0
        
        # === DONCHIAN BREAKOUT (Trend Follow) ===
        donchian_breakout_long = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
        donchian_breakout_short = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
        
        # === VOL FILTER ===
        vol_ratio = atr_14[i] / (atr_median + 1e-10)
        if vol_ratio > 2.5:
            position_size = BASE_SIZE * 0.5
        elif vol_ratio > 1.5:
            position_size = BASE_SIZE * 0.75
        else:
            position_size = BASE_SIZE
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        signal_strength = 0
        
        # === REGIME 1: CHOPPY/RANGE (CHOP > 61.8) — MEAN REVERSION + FUNDING ===
        if regime_chop:
            # Long: CRSI oversold + funding extreme negative (crowd too short)
            if crsi_oversold:
                signal_strength = 1
                if crsi_extreme_oversold:
                    signal_strength = 2
                if funding_extreme_long:
                    signal_strength += 2
                elif funding_moderate_long:
                    signal_strength += 1
                if price_above_hma_1w:
                    signal_strength += 1
                
                if signal_strength >= 2:
                    desired_signal = position_size * (0.8 + 0.2 * min(signal_strength, 5) / 5)
            
            # Short: CRSI overbought + funding extreme positive (crowd too long)
            if crsi_overbought and desired_signal == 0:
                signal_strength = 1
                if crsi_extreme_overbought:
                    signal_strength = 2
                if funding_extreme_short:
                    signal_strength += 2
                elif funding_moderate_short:
                    signal_strength += 1
                if price_below_hma_1w:
                    signal_strength += 1
                
                if signal_strength >= 2:
                    desired_signal = -position_size * (0.8 + 0.2 * min(signal_strength, 5) / 5)
        
        # === REGIME 2: TRENDING (CHOP < 38.2) — TREND FOLLOW + FUNDING CONFIRM ===
        elif regime_trend:
            # Long: Donchian breakout + HTF bullish + funding not extreme short
            if donchian_breakout_long:
                signal_strength = 1
                if price_above_hma_1w:
                    signal_strength += 2
                if hma_bullish:
                    signal_strength += 1
                if not funding_extreme_short:
                    signal_strength += 1
                
                if signal_strength >= 3:
                    desired_signal = position_size * (0.8 + 0.2 * min(signal_strength, 5) / 5)
            
            # Short: Donchian breakdown + HTF bearish + funding not extreme long
            if donchian_breakout_short and desired_signal == 0:
                signal_strength = 1
                if price_below_hma_1w:
                    signal_strength += 2
                if hma_bearish:
                    signal_strength += 1
                if not funding_extreme_long:
                    signal_strength += 1
                
                if signal_strength >= 3:
                    desired_signal = -position_size * (0.8 + 0.2 * min(signal_strength, 5) / 5)
        
        # === REGIME 3: TRANSITION (38.2-61.8) — FUNDING CONTRARIAN PRIMARY ===
        else:
            # Funding extreme is primary signal in transition
            if funding_extreme_long and not price_below_hma_1w:
                desired_signal = position_size * 0.6
            elif funding_extreme_short and not price_above_hma_1w:
                desired_signal = -position_size * 0.6
            elif crsi_extreme_oversold and not price_below_hma_1w:
                desired_signal = position_size * 0.5
            elif crsi_extreme_overbought and not price_above_hma_1w:
                desired_signal = -position_size * 0.5
        
        # === CAP SIGNAL TO MAX 0.35 ===
        if desired_signal > 0.35:
            desired_signal = 0.35
        elif desired_signal < -0.35:
            desired_signal = -0.35
        
        # === STOPLOSS CHECK (Trailing ATR) ===
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
        
        # === CRSI EXTREME EXIT (Take Profit) ===
        if in_position and position_side > 0 and crsi[i] > 80.0:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and crsi[i] < 20.0:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if bias unchanged ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0 and (hma_bullish or price_above_hma_1w):
                desired_signal = position_size
            elif position_side < 0 and (hma_bearish or price_below_hma_1w):
                desired_signal = -position_size
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal != 0.0:
            if desired_signal > 0:
                if desired_signal >= 0.28:
                    desired_signal = 0.30
                elif desired_signal >= 0.22:
                    desired_signal = 0.25
                else:
                    desired_signal = 0.15
            else:
                if desired_signal <= -0.28:
                    desired_signal = -0.30
                elif desired_signal <= -0.22:
                    desired_signal = -0.25
                else:
                    desired_signal = -0.15
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
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
2026-03-23 10:46
